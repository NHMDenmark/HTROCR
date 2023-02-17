import os


if __name__ == '__main__':
    # Prepare line level images
    path = "../../data/training_data/emunch"

    for root, dirs, files in os.walk(path):
        for f in files:
            if ("gt_train" in f) or ("_line_" in f) or ("questionable" in f):
                rpath = os.path.join(root, f)
                os.remove(rpath)