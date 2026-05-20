import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, LineString
import editdistance

def poly_center(poly_pts):
    """Calculate center point from polygon"""
    poly_pts = np.array(poly_pts).reshape(-1, 2)
    num_points = poly_pts.shape[0]
    if num_points >= 4:
        line1 = LineString(poly_pts[int(num_points/2):])
        line2 = LineString(poly_pts[:int(num_points/2)])
        mid_pt1 = np.array(line1.interpolate(0.5, normalized=True).coords[0])
        mid_pt2 = np.array(line2.interpolate(0.5, normalized=True).coords[0])
        return (mid_pt1 + mid_pt2) / 2
    elif num_points >= 2:
        return np.mean(poly_pts, axis=0)
    else:
        return poly_pts[0] if len(poly_pts) > 0 else np.array([0, 0])

def decode_recognition(output_seq, chars, start_index=1000, max_length=25):
    """Decode recognition from model output - matching test script logic"""
    text = ''
    for c in output_seq:
        if start_index <= c < start_index + len(chars):
            text += chars[c - start_index]
        else:
            break
    return text

def read_ground_truth(gt_folder):
    """Read ground truth files for ICDAR 2019 format"""
    gts = {}
    gt_files = glob.glob(f"{gt_folder}/*.txt")
    gt_files.sort()
    
    for gt_file in gt_files:
        # Extract image ID from filename (e.g., tr_img_00004.txt -> 4)
        filename = os.path.basename(gt_file)
        if 'tr_img_' in filename:
            image_id = int(filename.replace('tr_img_', '').replace('.txt', ''))
        elif 'img_' in filename:
            image_id = int(filename.replace('img_', '').replace('.txt', ''))
        else:
            image_id = int(os.path.splitext(filename)[0])
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        points = []
        texts = []
        languages = []
        dontcares = []
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.strip().split(',')
            
            # ICDAR 2019 format
            if len(parts) >= 10:
                coords = [int(float(x)) for x in parts[:8]]
                language = parts[8]
                text = parts[9]
                
                # Reshape to 4 points
                polygon_pts = [
                    [coords[0], coords[1]],
                    [coords[2], coords[3]],
                    [coords[4], coords[5]],
                    [coords[6], coords[7]]
                ]
                
                # Calculate center
                center = poly_center(polygon_pts)
                center_point = Point(center[0], center[1])
                
                # Don't care regions
                dontcare = (text == "###" or text == "ignore" or text == "")
                
                points.append(center_point)
                texts.append(text)
                languages.append(language)
                dontcares.append(dontcare)
        
        gts[image_id] = {
            'points': points,
            'texts': texts,
            'languages': languages,
            'matched_det': [0] * len(points),
            'matched_e2e': [0] * len(points),
            'dontcares': dontcares
        }
    
    return gts

def evaluate_detection(pred_points, gt_points, distance_threshold=5.0):
    """Match predictions to GT based on point distance"""
    if not pred_points or not gt_points:
        return [], []
    
    matched_gt = []
    matched_pred = []
    
    # For each prediction, find closest GT
    for pred_idx, pred_point in enumerate(pred_points):
        min_dist = float('inf')
        best_gt_idx = -1
        
        for gt_idx, gt_point in enumerate(gt_points):
            dist = pred_point.distance(gt_point)
            if dist < min_dist:
                min_dist = dist
                best_gt_idx = gt_idx
        
        # Match if within threshold and GT not already matched
        if min_dist <= distance_threshold and best_gt_idx not in matched_gt:
            matched_gt.append(best_gt_idx)
            matched_pred.append(pred_idx)
    
    return matched_gt, matched_pred

def evaluate_all(results, gts, det_threshold=5.0, conf_threshold=0.922):
    """Complete evaluation with detection and recognition"""
    
    # Group predictions by image
    pred_by_image = {}
    for pred in results:
        img_id = pred['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(pred)
    
    # Statistics
    total_tp_det = 0  # True positives for detection
    total_fp_det = 0  # False positives for detection
    total_fn_det = 0  # False negatives for detection
    
    total_correct_rec = 0  # Correct recognition on matched detections
    total_matched_det = 0   # Total matched detections
    
    total_tp_e2e = 0  # True positives for end-to-end
    
    # Process each image
    for img_id, preds in pred_by_image.items():
        if img_id not in gts:
            total_fp_det += len([p for p in preds if p.get('score', 0) >= conf_threshold])
            continue
        
        gt = gts[img_id]
        
        # Filter by confidence threshold (using the same 0.922 from test script)
        valid_preds = [p for p in preds if p.get('score', 0) >= conf_threshold]
        
        if not valid_preds:
            # Count false negatives
            valid_gt_count = sum(1 for i, dc in enumerate(gt['dontcares']) if not dc)
            total_fn_det += valid_gt_count
            continue
        
        # Create prediction points
        pred_points = []
        for pred in valid_preds:
            polys = pred.get('polys', [])
            if polys:
                if len(polys) == 1:
                    pred_points.append(Point(polys[0][0], polys[0][1]))
                else:
                    center = poly_center(polys)
                    pred_points.append(Point(center[0], center[1]))
            else:
                pred_points.append(Point(0, 0))
        
        # Match detections
        matched_gt, matched_pred = evaluate_detection(pred_points, gt['points'], det_threshold)
        
        # Detection metrics
        for gt_idx in matched_gt:
            if not gt['dontcares'][gt_idx]:
                total_tp_det += 1
        
        total_fp_det += len(valid_preds) - len(matched_pred)
        
        valid_gt_count = sum(1 for i, dc in enumerate(gt['dontcares']) if not dc)
        total_fn_det += valid_gt_count - len(matched_gt)
        
        # Recognition metrics
        for gt_idx, pred_idx in zip(matched_gt, matched_pred):
            if gt['dontcares'][gt_idx]:
                continue
            
            pred_text = valid_preds[pred_idx].get('rec', '').strip()
            gt_text = gt['texts'][gt_idx]
            
            total_matched_det += 1
            
            if pred_text and pred_text.upper() == gt_text.upper():
                total_correct_rec += 1
                total_tp_e2e += 1  # End-to-end correct only if recognition matches
    
    # Calculate metrics
    det_precision = total_tp_det / (total_tp_det + total_fp_det) if (total_tp_det + total_fp_det) > 0 else 0
    det_recall = total_tp_det / (total_tp_det + total_fn_det) if (total_tp_det + total_fn_det) > 0 else 0
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0
    
    rec_accuracy = total_correct_rec / total_matched_det if total_matched_det > 0 else 0
    
    e2e_precision = total_tp_e2e / (total_tp_det + total_fp_det) if (total_tp_det + total_fp_det) > 0 else 0
    e2e_recall = total_tp_e2e / (total_tp_e2e + total_fn_det) if (total_tp_e2e + total_fn_det) > 0 else 0
    e2e_f1 = 2 * e2e_precision * e2e_recall / (e2e_precision + e2e_recall) if (e2e_precision + e2e_recall) > 0 else 0
    
    return {
        'detection': {'precision': det_precision, 'recall': det_recall, 'f1': det_f1,
                     'tp': total_tp_det, 'fp': total_fp_det, 'fn': total_fn_det},
        'recognition': {'accuracy': rec_accuracy, 'matched_detections': total_matched_det,
                       'correct': total_correct_rec},
        'end_to_end': {'precision': e2e_precision, 'recall': e2e_recall, 'f1': e2e_f1,
                      'tp': total_tp_e2e}
    }

def fix_predictions_recognition(result_path, chars, start_index=1000, max_length=25, min_confidence=0.922):
    """Fix predictions by re-decoding recognition using test script logic"""
    print("\n" + "="*50)
    print("FIXING PREDICTIONS - RE-DECODING RECOGNITION")
    print("="*50)
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    print(f"Original predictions: {len(results)}")
    
    fixed_count = 0
    for pred in results:
        # Check if recognition is empty but value array exists
        if not pred.get('rec', '') and pred.get('value'):
            # Re-decode from value array (which contains the raw outputs)
            # The value array contains [x, y, char1, char2, ...]
            values = pred.get('value', [])
            if len(values) > 2:
                # Extract character indices (after x,y)
                char_indices = values[2:]
                # Decode using test script logic
                text = ''
                for c in char_indices:
                    c_int = int(c) if isinstance(c, (int, float)) else c
                    if start_index <= c_int < start_index + len(chars):
                        text += chars[c_int - start_index]
                    else:
                        break
                
                if text:
                    pred['rec'] = text
                    fixed_count += 1
                    if fixed_count <= 5:  # Print first few fixes
                        print(f"  Fixed: '{text}' (confidence: {pred.get('score', 0):.3f})")
    
    print(f"Fixed {fixed_count} predictions with empty recognition")
    
    # Save fixed predictions
    fixed_path = result_path.replace('.json', '_fixed.json')
    with open(fixed_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved fixed predictions to: {fixed_path}")
    
    return results, fixed_path

def analyze_predictions(result_path):
    """Analyze predictions to understand what's there"""
    print("\n" + "="*50)
    print("ANALYZING PREDICTIONS")
    print("="*50)
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    print(f"Total predictions: {len(results)}")
    
    # Check recognition field
    empty_rec = sum(1 for r in results if not r.get('rec', ''))
    print(f"Empty recognition: {empty_rec} ({empty_rec/len(results)*100:.1f}%)")
    
    # Check value array (raw outputs)
    has_value = sum(1 for r in results if r.get('value'))
    print(f"Has value array: {has_value} ({has_value/len(results)*100:.1f}%)")
    
    # Sample predictions
    print("\nSample predictions (first 5):")
    for i, pred in enumerate(results[:5]):
        print(f"\nPred {i+1}:")
        print(f"  Image ID: {pred.get('image_id')}")
        print(f"  Score: {pred.get('score', 0):.4f}")
        print(f"  Rec: '{pred.get('rec', '')}'")
        print(f"  Polys: {pred.get('polys', [])}")
        if pred.get('value'):
            print(f"  Value (first 10): {pred['value'][:10]}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True,
                       help='Path to JSON results from model')
    parser.add_argument('--gt_folder', type=str, required=True,
                       help='Path to ground truth folder')
    parser.add_argument('--det_threshold', type=float, default=5.0,
                       help='Distance threshold for detection (pixels)')
    parser.add_argument('--conf_threshold', type=float, default=0.922,
                       help='Confidence threshold (default 0.922 from test script)')
    parser.add_argument('--chars', type=str, 
                       default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~',
                       help='Character set for decoding')
    parser.add_argument('--fix_recognition', action='store_true',
                       help='Try to fix empty recognition by re-decoding from value array')
    args = parser.parse_args()
    
    # First analyze predictions
    results = analyze_predictions(args.result_path)
    
    # Fix recognition if requested
    if args.fix_recognition:
        results, fixed_path = fix_predictions_recognition(
            args.result_path, args.chars, start_index=1000, 
            min_confidence=args.conf_threshold
        )
        result_path_to_use = fixed_path
    else:
        result_path_to_use = args.result_path
    
    # Load ground truth
    print("\n" + "="*50)
    print("LOADING GROUND TRUTH")
    print("="*50)
    gts = read_ground_truth(args.gt_folder)
    print(f"Loaded {len(gts)} images with ground truth")
    
    total_gt_anns = sum(len(gt['points']) for gt in gts.values())
    print(f"Total GT annotations: {total_gt_anns}")
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    metrics = evaluate_all(results, gts, args.det_threshold, args.conf_threshold)
    
    print("\n" + "="*50)
    print("DETECTION METRICS")
    print("="*50)
    print(f"Precision: {metrics['detection']['precision']:.4f}")
    print(f"Recall: {metrics['detection']['recall']:.4f}")
    print(f"F1-Score: {metrics['detection']['f1']:.4f}")
    print(f"TP: {metrics['detection']['tp']}, FP: {metrics['detection']['fp']}, FN: {metrics['detection']['fn']}")
    
    print("\n" + "="*50)
    print("RECOGNITION METRICS (on matched detections)")
    print("="*50)
    print(f"Accuracy: {metrics['recognition']['accuracy']:.4f}")
    print(f"Matched detections: {metrics['recognition']['matched_detections']}")
    print(f"Correct recognitions: {metrics['recognition']['correct']}")
    
    print("\n" + "="*50)
    print("END-TO-END METRICS")
    print("="*50)
    print(f"Precision: {metrics['end_to_end']['precision']:.4f}")
    print(f"Recall: {metrics['end_to_end']['recall']:.4f}")
    print(f"F1-Score: {metrics['end_to_end']['f1']:.4f}")

if __name__ == '__main__':
    main()