import time

from NER.NER2016withPytorch.evaluating import Metrics
from NER.NER2016withPytorch.models.bilstm_crf import BILSTM_Model
from NER.named_entity_recognition.utils import save_model


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word_to_id, tag_to_id, crf=True, remove_O=False):
    train_word_lists = [train_data[i]['words'] for i in range(len(train_data))]
    train_tag_lists = [train_data[i]['tags'] for i in range(len(train_data))]
    dev_word_lists = [dev_data[i]['words'] for i in range(len(dev_data))]
    dev_tag_lists = [dev_data[i]['tags'] for i in range(len(dev_data))]
    test_word_lists = [test_data[i]['words'] for i in range(len(test_data))]
    test_tag_lists = [test_data[i]['tags'] for i in range(len(test_data))]

    start = time.time()
    vocab_size = len(word_to_id)
    out_size = len(tag_to_id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word_to_id, tag_to_id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word_to_id, tag_to_id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists