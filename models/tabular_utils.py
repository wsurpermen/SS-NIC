import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

class Transformer:

    @staticmethod
    def get_metadata(data,c_meta, categorical_columns=tuple(), ordinal_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            if index in categorical_columns:
                mapper = column.value_counts(sort=False).index.tolist()
                meta.append({
                    "name": c_meta[index]['name'],
                    "type": CATEGORICAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            elif index in ordinal_columns:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
                meta.append({
                    "name":  c_meta[index]['name'],
                    "type": ORDINAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            else: 
                meta.append({
                    "name":  c_meta[index]['name'],
                    "type": CONTINUOUS,
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError


class GeneralTransformer(Transformer):
    """Continuous and ordinal columns are normalized to [0, 1].
    Discrete columns are converted to a one-hot vector.
    """

    def __init__(self, encoder , act='sigmoid'):
        self.act = act
        self.meta = None
        self.output_dim = None
        self.encoder = encoder

    def fit(self, data,c_meta, categorical_columns=tuple(), ordinal_columns=tuple()):
    # def fit(self, data, meta):
        #self.meta = self.get_metadata(data,c_meta,categorical_columns, ordinal_columns)
        self.meta = c_meta
        self.output_dim = 0
        for info in self.meta:
            if info['type'] in [CONTINUOUS, ORDINAL]:
                self.output_dim += 1
            else:
                self.output_dim += info['size']

    def transform(self, data):
        data_t = []
        self.output_info = []
        ordinal_e = lambda col, info: (col / info['size'] * 2 - 1) if self.act == 'tanh' else col / info['size']
        for id_, info in enumerate(self.meta):
            col = data[:, id_]
            if info['type'] == CONTINUOUS:
                col = col.astype(np.float64)
                col[np.isnan(col)] = np.median(col[~np.isnan(col)])
                if not (info['min']==info['max']):
                    col = (col - (info['min'])) / (info['max'] - info['min'])
                else:
                    col -= info['min']
                if self.act == 'tanh':
                    col = col * 2 - 1
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, self.act))

            elif info['type'] == ORDINAL:
                data_t.append(ordinal_e(col,info).reshape([-1,1]))
                self.output_info.append((1, self.act))

            # elif info['name'] != 'label':
            else:
                if self.encoder == 'ordinal':
                    data_t.append(ordinal_e(col, info).reshape([-1, 1]))
                    self.output_info.append((1, self.act))
                    continue
                col_t = np.zeros([len(data), info['size']])

                idx = list(map(info['i2s'].index, col))
                col_t[np.arange(len(data)), idx] = 1
                data_t.append(col_t)
                self.output_info.append((info['size'], 'softmax'))
            # else: # for label
            #     col_t = np.array(list(map(info['i2s'].index, col)))
            #     data_t.append(col_t.reshape(-1,1))
            #     self.output_info.append((1, 'softmax'))

        return np.concatenate(data_t, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)], dtype=object)

        data = data.copy()
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                current = data[:, 0]
                data = data[:, 1:]

                if self.act == 'tanh':
                    current = (current + 1) / 2

                current = np.clip(current, 0, 1)
                data_t[:, id_] = current * (info['max'] - info['min']) + info['min']

            elif info['type'] == ORDINAL:
                current = data[:, 0]
                data = data[:, 1:]

                if self.act == 'tanh':
                    current = (current + 1) / 2

                current = current * info['size']
                current = np.round(current).clip(0, info['size'] - 1)
                data_t[:, id_] = current
            # elif info['name'] != 'label':
            else:
                if self.encoder =='ordinal':
                    current = data[:, 0]
                    data = data[:, 1:]

                    if self.act == 'tanh':
                        current = (current + 1) / 2

                    current = current * info['size']
                    current = np.round(current).clip(0, info['size'] - 1)
                    data_t[:, id_] = current
                    continue

                current = data[:, :info['size']]
                data = data[:, info['size']:]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))
            # else: # map the label, and the label should be the last col
            #
            #     current = np.array(data[:, :info['size']]).reshape(-1).astype(np.int32)
            #     data_t[:, id_] = list(map(info['i2s'].__getitem__, current))

        return data_t
