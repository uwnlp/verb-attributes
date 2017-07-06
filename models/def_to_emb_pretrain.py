"""
This model pretrains the BLSTM to embedding model through dictionary data.
"""

from config import DICTIONARY_PATH

data = DictLoader()
print("everything set up!")

m_train = DictionaryModel(data, config)
m_val = DictionaryModel.from_model(m_train, mode='val')

last_best = 0
with tf.Session(graph=m_train.graph) as sess:
    coord = tf.train.Coordinator()
    m_train.restore(sess)
    m_val.restore(sess)

    prev_best_val = m_val.val_epoch(sess)

    for epoch in range(1, config.num_epochs+1):
        for batch in range(1, data.num_train_batches+1):
            log_summary = batch % 100 == 0
            loss = m_train.train_update(sess, log_summary=log_summary)

        m_train.reset_metrics(sess)
        l = m_val.val_epoch(sess)
        if l < prev_best_val:
            print("New best")
            prev_best_val = l
            m_train.save(sess)
            last_best = epoch
        else:
            print("Not saving!!")
        if last_best < epoch-2:
            print("Early stopping at epoch {}".format(epoch))
            break
    #
    # m_test = DictionaryModel.from_model(m_train, mode='test')
    # m_test.restore(sess)
    # print("Testing!")
    # m_test.test_epoch(sess, save_overall='{}_overall.csv'.format(config.name), save_everything='{}_everything.csv'.format(config.name))