import glob
import pickle


def reformat_predictions(pickle_path):
    with open(pickle_path, 'rb') as f:
        preds = pickle.load(f)
    assert len(preds['filename']) == len(preds['predictions'])
    reformatted = {fp.split('/')[-1].split('.')[0] + '_image' : p[0] for fp, p in zip(preds['filename'], preds['predictions'])}
    return reformatted


pickled_preds = glob.glob('../predictions/swin00*_fold*pkl')

for pp in pickled_preds:
    reformatted = reformat_predictions(pp)
    with open(pp.replace('.pkl', '_reformat.pkl'), 'wb') as f:
        pickle.dump(reformatted, f)


