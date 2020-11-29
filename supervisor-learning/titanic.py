import tensorflow as tf
import pandas as pd
import numpy as np

import os


def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    #age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp, Parch, Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Cabin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return dfresult.values


def build_graph(input_dim):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def fc(in_shape, out_shape, input, name_space):
        with tf.name_scope(name_space):
            w_fc = weight_variable([in_shape, out_shape])
            b_fc = bias_variable([out_shape])
            output_fc = tf.matmul(input, w_fc) + b_fc
            return output_fc

    x = tf.placeholder(tf.float32, [None, input_dim], name='x-input')
    y_ = tf.placeholder(tf.int64, [None, ], name='y-input')

    out_fc1 = tf.nn.relu(fc(input_dim, 20, x, 'fc1'))
    out_fc2 = tf.nn.relu(fc(20, 10, out_fc1, 'fc2'))

    y = tf.nn.softmax(fc(10, 2, out_fc2, 'fc3'), name='softmax-output')

    pred  = tf.argmax(y, 1, name='pred')

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    correct_prediction = tf.equal(pred, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return {'x_input': x, 'y_input': y_, 'softmax-output': y, 'pred': pred, 'loss': loss,
            'optimization': opt, 'accuracy': accuracy}


bz = 64
lr = 0.01
epochs = 30

if __name__ == '__main__':

    dftrain_raw = pd.read_csv('../data/titanic/train.csv')
    dftest_raw = pd.read_csv('../data/titanic/test.csv')
    x_train = preprocessing(dftrain_raw)
    x_test = preprocessing(dftest_raw)
    y_train = dftrain_raw['Survived'].values
    y_test = dftest_raw['Survived'].values
    train_size = x_train.shape[0]
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.batch(bz)
    iterator = dataset_train.make_initializable_iterator()
    features, labels = iterator.get_next()

    graph = build_graph(15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            sess.run(iterator.initializer)
            i = 0
            while 1:
                feature, label = sess.run([features, labels])
                print(label.shape)
                _, acc, loss = sess.run([graph['optimization'], graph['accuracy'], graph['loss']],
                     feed_dict={graph['x_input']: feature, graph['y_input']: label})
                print("epoch [{:>2}/{}], iter [{:>2}/{}], loss {:.4f}, accuracy {:.4f}".
                      format(epoch, epochs, i, train_size // bz, loss, acc))
                i += 1
                if i % (train_size // bz) == 0 and i > 0:
                    break

        acc_test = sess.run(graph['accuracy'], feed_dict={graph['x_input']: x_test, graph['y_input']: y_test})
        print("test accuracy: {:.4f}".format(acc_test))

        # for i in range(10):
        #     f, l = sess.run([features, labels])
        #     print(i, l)
        #     i += 1
        #     if i % (train_size // bz) == 0 and i > 0:
        #         sess.run(iterator.initializer)