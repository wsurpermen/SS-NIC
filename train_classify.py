import numpy as np
import logging
from models import ncsnpp_tabular
import losses
from models.ema import ExponentialMovingAverage
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score,roc_curve
from sklearn.model_selection import KFold
import datasets
from torch.utils.data import DataLoader
import sde_lib
from absl import flags
from models.utils import add_label_noise
import torch
from torch.utils import tensorboard
from utils import save_checkpoint, restore_checkpoint, apply_activate
import os
from models import classify_net
import torch.nn.functional as F
import random
from torch.utils.data import RandomSampler
from fineturne import clean_fn
'''
    trained with data_without_labels;
'''

FLAGS = flags.FLAGS
def train_step_fn(state, batch, sde, labels, optimize_fn, eps = 1e-5):
    # labels is one-hot,
    model = state['model']
    optimizer = state['optimizer']
    optimizer.zero_grad()
    loss_criterion = 'ce'
    # diffuse the data
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None] * z
    # fed the model
    predict = model(perturbed_data, sde.marginal_prob(torch.zeros_like(perturbed_data), t)[1])
    if loss_criterion == 'mae':
        loss = torch.mean(torch.abs( F.softmax(predict, dim=1) - labels.float()))
    elif loss_criterion == 'ce':
        loss = F.cross_entropy(predict, torch.argmax(labels,dim=1))
        loss = loss.mean()
    elif loss_criterion == 'sce':
        alpha=1.0
        beta=1.0
        ce_loss = F.cross_entropy(predict, torch.argmax(labels,dim=1))

        # Convert labels to one-hot encoding

        # Compute probabilities with softmax
        probs = F.softmax(predict, dim=1)

        # Compute Reverse Cross-Entropy (RCE) loss
        rce_loss = -torch.sum(probs * torch.log(labels.float() + 1e-5), dim=1).mean()

        # Combine CE and RCE losses
        loss = alpha * ce_loss + beta * rce_loss
    elif loss_criterion == 'gce':
        q=0.7
        pred = F.softmax(predict, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        loss = ((1. - torch.pow(torch.sum(labels.float() * pred, dim=1), q)) / q).mean()
    else:
        raise ValueError('no such loss criterion')
    loss.backward()
    optimize_fn(optimizer, model.parameters(), step=state['step'])
    state['step'] += 1
    state['ema'].update(model.parameters())
    state['ema'].copy_to(model.parameters())
    return loss

def load_labels(path):
    c_path = os.path.join(os.path.dirname(__file__), 'tabular_datasets')
    local_path = os.path.join(c_path, path + '.npz')
    return local_path



def train(config, workdir):


    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)



    # Build data iterators
    train_ds, eval_ds, (transformer, meta) = datasets.get_dataset(config,
                                                                  uniform_dequantization=config.data.uniform_dequantization)
    data = np.concatenate((train_ds, eval_ds), axis=0)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    ckp_fdix = 0

    for train_index, val_index in kf.split(data):
        randomSeed = 2021
        torch.manual_seed(randomSeed)
        torch.cuda.manual_seed(randomSeed)
        torch.cuda.manual_seed_all(randomSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(randomSeed)

        ckp_fdix += 1
        logging.info(f"fold {ckp_fdix}")
        train_ds, eval_ds = data[train_index], data[val_index]
        # add noise
        label = train_ds[:, -config.data.output_size:]
        noise, _ = add_label_noise(label, config)
        train_ds[:, -config.data.output_size:] = np.array(noise)

        test_labels = np.array(noise)
        test_labels = np.argmax(test_labels, axis=1)
        test_eval_labels = np.argmax(eval_ds[:, -config.data.output_size:], axis=1)
        eval_ds = eval_ds[:, :-config.data.output_size]
        eval_ds = torch.tensor(eval_ds).to(config.device).float()

        cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1, average="macro"),
        recall_score(l1, p1, average="macro"), f1_score(l1, p1, average="macro"))

        # Initialize model.
        class_model = torch.nn.DataParallel(classify_net.NCSNpp(config).to(config.device))
        num_params = sum(p.numel() for p in class_model.parameters())

        ema = ExponentialMovingAverage(class_model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, class_model.parameters())
        state = dict(optimizer=optimizer, model=class_model, ema=ema, step=0, epoch=0)

        checkpoint_dir = os.path.join(workdir, "checkpoints")
        checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", f"checkpoint{ckp_fdix}.pth")

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
        initial_step = int(state['epoch'])

        logging.info(f"train shape : {train_ds.shape}")
        logging.info(f"eval.shape : {eval_ds.shape}")

        logging.info(f"batch size: {config.training.batch_size}")

        #sampler = RandomSampler(train_ds, replacement=True, num_samples=((len(train_ds) // 1000) + 1) * 1000)
        train_iter = DataLoader(train_ds, batch_size=config.training.batch_size, pin_memory=True)

        logging.info(class_model)

        logging.info("Starting training loop at epoch %d." % (initial_step,))

        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        elif config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                   N=config.model.num_scales)
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                N=config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
        optimize_fn = losses.optimization_manager(config)
        max_acc = 0
        for epoch in range(initial_step, config.training.cls_epoch + 1):
        #for epoch in range(initial_step, 10):

            state['epoch'] += 1
            for iteration, batch in enumerate(train_iter):
                train_labels = batch[:, -config.data.output_size:].to(config.device)

                batch = batch[:, :-config.data.output_size].to(config.device).float()
                # we need a labels to calculate the loss
                loss = train_step_fn(state, batch, sde, train_labels, optimize_fn)
                writer.add_scalar("training_loss", loss, state['step'])
            logging.info("epoch: %d, training_loss: %.5e" % (epoch, loss))
            if epoch % 50 == 0:
                save_checkpoint(checkpoint_meta_dir, state)  # save the model each epoch

            if epoch % 10 ==0:
                with torch.no_grad():
                    model = state['model']
                    if isinstance(model, torch.nn.DataParallel):  # for DataParaller
                        model = model.module
                    p = model(torch.tensor(train_ds[:, :-config.data.output_size]).to(config.device).float(),
                              torch.tensor([1e-5]).to(config.device)).cpu()

                    # p = model(eval_ds, torch.tensor([1e-5]).to(config.device))
                    predict = torch.argmax(p, dim=1).cpu()
                    predict = predict.detach().numpy()

                # task the predit accurate
                # print(test_labels)
                # print(predict)
                acc, precision, recall, f1 = cal_metric(test_labels, predict)
                writer.add_scalar("acc", acc, state['step'])
                # writer.add_scalar("precision", precision, state['step'])
                writer.add_scalar("recall", recall, state['step'])
                writer.add_scalar("f1", f1, state['step'])
                logging.info("epoch: %d, iter: %d, acc: %.5e" % (epoch, iteration, acc))
                logging.info("epoch: %d, iter: %d, precision: %.5e" % (epoch, iteration, precision))
                logging.info("epoch: %d, iter: %d, recall: %.5e" % (epoch, iteration, recall))
                logging.info("epoch: %d, iter: %d, f1: %.5e" % (epoch, iteration, f1))


                if max_acc < acc:
                    max_acc = acc
                    save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_max{ckp_fdix}.pth'), state)

            if 0 and config.training.label:
                cleaner_data = clean_fn(flags.workdir.split('_')[0], config, train_ds, ckp_fdix)
                train_iter = DataLoader(cleaner_data, batch_size=config.training.batch_size, pin_memory=True)




        ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_max{ckp_fdix}.pth")
        state = restore_checkpoint(ckpt_filename, state, device=config.device)
        model = state['model']


        if isinstance(model, torch.nn.DataParallel):  # for DataParaller
            model = model.module

        with torch.no_grad():
            p = model(eval_ds, torch.tensor([1e-5]).to(config.device)).cpu()
        predict = torch.argmax(p,dim=1).cpu()
        predict = predict.detach().numpy()
        # predict_score = [p[i][predict[i]] for i in range(len(predict))]
        # print(predict)
        # fpr, tpr, thresholds = roc_curve(predict, predict_score)

        # auc1 = auc(fpr, tpr)
        acc, precision, recall, f1 = cal_metric(test_eval_labels, predict)


        logging.info("finally, the number of parameters{}".format(num_params))
        logging.info('finally, acc:{}'.format(acc))
        # logging.info('auc1:{}'.format(auc1))
        logging.info('finally, precision:{}'.format(precision))
        logging.info('finally, recall:{}'.format(recall))
        logging.info('finally, f1:{}'.format(f1))
