import torchmetrics
import argparse
import os
import math
import numpy as np

RESULTS_DIR = '/Users/linas/Studies/UCPH-DIKU/thesis/code/data/single_line_results'
GT_DIR = 'NHMD_GT'

def get_scores(data_dir):
    dir_loc = os.path.join(RESULTS_DIR, data_dir)
    cers = []
    wers = []
    cer = torchmetrics.CharErrorRate()
    wer = torchmetrics.WordErrorRate()
    for file in os.listdir(dir_loc):
        file_path = os.path.join(dir_loc, file)
        gt_path = os.path.join(RESULTS_DIR, GT_DIR, file)
        with open(file_path) as r:
            output = r.read()
        with open(gt_path) as r:
            gt_transcriptions = r.read()
        c = np.clip(float(cer(output.lower(), gt_transcriptions.lower()).numpy()), 0.0, 1.0)
        w = np.clip(float(wer(output.lower(), gt_transcriptions.lower()).numpy()), 0.0, 1.0)
        # Handling the case where no text was transcribed at all
        if math.isnan(c) or math.isnan(w):
            c = 1.0
            w = 1.0
        cers.append(c)
        wers.append(w)
    # Computing a mean per each paragraph to reduce overall error rate
    # caused by inaccurate layout analysis
    cer = np.array(cers).mean()
    wer = np.array(wers).mean()
    return cer, wer

def gen_response(cer_bbox, wer_bbox, cer_orig, wer_orig):
    res_str = 'BBOX CER: {}\nBBOX WER: {}\n'.format(cer_bbox, wer_bbox)
    res_str += 'FULLPAGE CER: {}\nFULLPAGE WER: {}\n'.format(cer_orig, wer_orig)
    return res_str

def evaluate_model(args):
    response = ''
    if args.pylaia or args.all:
        # Getting bounding box results
        cer_bbox, wer_bbox = get_scores("PYLAIA_NHMD_BBOX_DANISH")
        # Getting original full image results
        cer_orig, wer_orig = get_scores("PYLAIA_NHMD_ORIG_DANISH")
        response += '===== PyLaia =====\n' 
        response += gen_response(cer_bbox, wer_bbox, cer_orig, wer_orig)
    
    if args.sfr or args.all:
        cer_bbox, wer_bbox = get_scores("SFR_NHMD_BBOX_ICDAR17")
        cer_orig, wer_orig = get_scores("SFR_NHMD_ORIG_ICDAR17")
        response += '===== Start, Follow, Read =====\n'
        response += gen_response(cer_bbox, wer_bbox, cer_orig, wer_orig)

    if args.gva or args.all:
        cer_bbox, wer_bbox = get_scores("GVA_NHMD_BBOX_MIX")
        cer_orig, wer_orig = get_scores("GVA_NHMD_ORIG_MIX")
        response += '===== Google VISION-API =====\n'
        response += gen_response(cer_bbox, wer_bbox, cer_orig, wer_orig)
    
    if args.trocr or args.all:
        # cer, wer = get_scores("TROCR_NHMD_LINES_IAM_BASE")
        # response += '===== TrOCR =====\n'
        # response += 'LINE CER: {}\nLINE WER: {}\n'.format(cer, wer)
        # cer, wer = get_scores("../trocr_results/small_v2/single_line")
        # response += '===== TrOCR_SMALL - NHMD =====\n'
        # response += 'LINE CER: {}\nLINE WER: {}\n'.format(cer, wer)
        cer, wer = get_scores("../trocr_results/large/single_line")
        response += '===== TrOCR_LARGE - NHMD =====\n'
        response += 'LINE CER: {}\nLINE WER: {}\n'.format(cer, wer)
    
    if args.van or args.all:
        cer_bbox, wer_bbox = get_scores("VAN_NHMD_BBOX_IAM")
        cer_bbox_read, wer_bbox_read = get_scores("VAN_NHMD_BBOX_READ")
        cer_orig, wer_orig = get_scores("VAN_NHMD_ORIG_IAM")
        response += '===== Vertical Attention Network =====\n'
        response += gen_response(cer_bbox, wer_bbox, cer_orig, wer_orig)
        response += 'BBOX (BASED ON READ) CER: {}\nBBOX (BASED ON READ) WER: {}\n'.format(cer_bbox_read, wer_bbox_read)
    
    if args.origaminet or args.all:
        pass

    return response

if __name__ == '__main__':
    print("Evaluating...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pylaia", action=argparse.BooleanOptionalAction, help="Evaluate PyLaia model based on Danish dataset")
    parser.add_argument("--sfr", action=argparse.BooleanOptionalAction, help="Evaluate Star, Follow, Read model")
    parser.add_argument("--gva", action=argparse.BooleanOptionalAction, help="Evaluate Google Vision-API model")
    parser.add_argument("--trocr", action=argparse.BooleanOptionalAction, help="Evaluate TrOCR model")
    parser.add_argument("--van", action=argparse.BooleanOptionalAction, help="Evaluate Vertical Attention Network model")
    parser.add_argument("--origaminet", action=argparse.BooleanOptionalAction, help="Evaluate OrigamiNet model")
    parser.add_argument("--all", action=argparse.BooleanOptionalAction, help="Evaluate all models")
    args = parser.parse_args()
    results = evaluate_model(args)
    print("Results:")
    print(results)
