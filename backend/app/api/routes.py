from flask import Blueprint, request, jsonify, send_file
import os
import sys
import cv2
import io
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from backend.app.services.prediction import Stage1DamageClassifier
from backend.app.services.locate import Stage2DamageLocalizationDetector

api_blueprint = Blueprint('api', __name__, url_prefix='/api')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../../../', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STAGE1_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../../', 'Middleware/models/stage1/mobilenetv3_canny_best.keras')
STAGE2_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../../', 'Middleware/models/stage2/yoloBest.pt')

stage1_classifier = Stage1DamageClassifier(STAGE1_MODEL_PATH)
stage2_detector = Stage2DamageLocalizationDetector(STAGE2_MODEL_PATH)

@api_blueprint.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'service_status': 'healthy',
        'stage1_model_loaded': stage1_classifier.model is not None,
        'stage2_model_loaded': stage2_detector.detection_network is not None
    }), 200

@api_blueprint.route('/predict', methods=['POST'])
def predict_damage():
    if 'image' not in request.files:
        return jsonify({'error_message': 'No image file provided in request'}), 400
    
    uploaded_image_file = request.files['image']
    
    if uploaded_image_file.filename == '':
        return jsonify({'error_message': 'No image file selected'}), 400
    
    saved_image_path = os.path.join(UPLOAD_FOLDER, uploaded_image_file.filename)
    uploaded_image_file.save(saved_image_path)
    
    try:
        stage1_classification_result = stage1_classifier.classify_damage_status(saved_image_path)
        
        if not stage1_classification_result['is_car_damaged']:
            return jsonify({
                'pipeline_status': 'completed_at_stage1',
                'stage1_result': stage1_classification_result,
                'stage2_result': None,
                'message': 'Image classified as not damaged - Stage 2 skipped'
            }), 200
        
        stage2_localization_result = stage2_detector.localize_damages_in_image(
            saved_image_path, 
            stage1_classification_result
        )
        
        return jsonify({
            'pipeline_status': 'completed_full_pipeline',
            'stage1_result': {
                'stage1_classification': stage1_classification_result['stage1_classification'],
                'damage_probability_score': stage1_classification_result['damage_probability_score']
            },
            'stage2_result': stage2_localization_result
        }), 200
    
    except Exception as error:
        return jsonify({'error_message': f'Processing error: {str(error)}'}), 500
    
    finally:
        if os.path.exists(saved_image_path):
            os.remove(saved_image_path)

@api_blueprint.route('/batch-predict', methods=['POST'])
def batch_predict_damage():
    request_data = request.get_json()
    
    if 'image_directory_path' not in request_data:
        return jsonify({'error_message': 'No image directory path provided'}), 400
    
    image_directory_path = request_data['image_directory_path']
    
    if not os.path.exists(image_directory_path):
        return jsonify({'error_message': f'Directory not found: {image_directory_path}'}), 400
    
    valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        os.path.join(image_directory_path, filename) 
        for filename in os.listdir(image_directory_path)
        if os.path.splitext(filename)[1].lower() in valid_image_extensions
    ]
    
    if not image_files:
        return jsonify({'error_message': 'No valid image files found in directory'}), 400
    
    try:
        batch_processing_results = []
        
        for image_file_path in image_files:
            stage1_classification_result = stage1_classifier.classify_damage_status(image_file_path)
            
            if stage1_classification_result['is_car_damaged']:
                stage2_localization_result = stage2_detector.localize_damages_in_image(
                    image_file_path,
                    stage1_classification_result
                )
            else:
                stage2_localization_result = {
                    'stage2_status': 'skipped',
                    'reason': 'Not damaged',
                    'damage_regions_count': 0
                }
            
            batch_processing_results.append({
                'image_filename': os.path.basename(image_file_path),
                'stage1_classification': stage1_classification_result['stage1_classification'],
                'damage_probability_score': stage1_classification_result['damage_probability_score'],
                'stage2_damage_regions_count': stage2_localization_result.get('damage_regions_count', 0)
            })
        
        return jsonify({
            'batch_processing_status': 'completed',
            'total_images_processed': len(batch_processing_results),
            'results': batch_processing_results
        }), 200
    
    except Exception as error:
        return jsonify({'error_message': f'Batch processing error: {str(error)}'}), 500

@api_blueprint.route('/detect-bbox', methods=['POST'])
def detect_bounding_boxes():
    if 'image' not in request.files:
        return jsonify({'error_message': 'No image file provided'}), 400
    
    uploaded_image_file = request.files['image']
    
    if uploaded_image_file.filename == '':
        return jsonify({'error_message': 'No image file selected'}), 400
    
    saved_image_path = os.path.join(UPLOAD_FOLDER, uploaded_image_file.filename)
    uploaded_image_file.save(saved_image_path)
    
    try:
        stage1_classification_result = stage1_classifier.classify_damage_status(saved_image_path)
        
        if not stage1_classification_result['is_car_damaged']:
            return jsonify({
                'bbox_detection_status': 'skipped',
                'stage1_classification': stage1_classification_result['stage1_classification'],
                'message': 'No damage detected - no bounding boxes to draw'
            }), 200
        
        stage2_localization_result = stage2_detector.localize_damages_in_image(
            saved_image_path,
            stage1_classification_result
        )
        
        bbox_visualization_image = draw_bboxes_on_image(
            stage1_classification_result['original_image'],
            stage2_localization_result['detected_damage_bboxes']
        )
        
        return send_file(
            io.BytesIO(bbox_visualization_image),
            mimetype='image/png',
            as_attachment=True,
            download_name='damage_localization_with_bboxes.png'
        ), 200
    
    except Exception as error:
        return jsonify({'error_message': f'BBox detection error: {str(error)}'}), 500
    
    finally:
        if os.path.exists(saved_image_path):
            os.remove(saved_image_path)

def draw_bboxes_on_image(image_array: np.ndarray, detected_damage_bboxes: list) -> bytes:
    visualization_image = image_array.copy()
    
    for bbox_info in detected_damage_bboxes:
        x_min = bbox_info['x_min']
        y_min = bbox_info['y_min']
        x_max = bbox_info['x_max']
        y_max = bbox_info['y_max']
        confidence_score = bbox_info['confidence']
        
        cv2.rectangle(visualization_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label_text = f"Damage: {confidence_score:.4f}"
        cv2.putText(
            visualization_image, 
            label_text, 
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
    
    success, encoded_visualization = cv2.imencode('.png', visualization_image)
    
    if not success:
        raise ValueError("Failed to encode image to PNG")
    
    return encoded_visualization.tobytes()
