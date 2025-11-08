import { ArrowRight, ArrowLeft, Loader2 } from "lucide-react"; 
import { useState } from "react"; 
import { getDamageTypes } from "../services/predectiveModels"; 
import { PIPELINE_API_BASE_URL } from "../../public/constants";
import { fileStore } from "../utils/fileStore";

const Classifier = ({ onBack, onNext }) => {
    const [loading, setLoading] = useState(false);
    const [segmentationResult, setSegmentationResult] = useState(null);
    const [error, setError] = useState(null);
    const image = fileStore.current;
    
    const handleAnalyze = async () => {
        if (!image) {
            setError('No image available. Please go back and upload an image.');
            return;
        }

        setLoading(true);
        setError(null);
        setSegmentationResult(null);

        try {
            const response = await getDamageTypes(image);
            const data = response.data;
            setSegmentationResult(data);
            console.log(data);
        } catch (error) {
            console.error('Segmentation error:', error);
            setError('Failed to analyze damage segmentation. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="bg-purple-50 border-2 border-purple-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-purple-800 mb-3">
                    Damage Segmentation Analysis
                </h2>
                <p className="text-purple-700">
                    Analyzing damaged areas using YOLOv9 segmentation model...
                </p>
            </div>

            {error && (
                <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4">
                    <p className="text-red-700">{error}</p>
                </div>
            )}

            <button
                onClick={handleAnalyze}
                className="w-full bg-purple-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-purple-600 transition-colors disabled:opacity-50 flex justify-center items-center gap-2"
                disabled={loading || !image}
            >
                {loading ? <Loader2 className="animate-spin" /> : null}
                {loading ? 'Analyzing...' : 'Run Segmentation Analysis'}
            </button>

            {segmentationResult?.result?.annotated_image && (
                <div className="bg-white border-2 border-gray-200 rounded-lg p-4">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">
                        Segmented Damage Areas
                    </h3>
                    <img 
                        src={`${PIPELINE_API_BASE_URL}${segmentationResult.result.annotated_image}`}
                        alt="Annotated damage segmentation"
                        className="w-full h-auto rounded-lg shadow-md"
                        onError={(e) => {
                            console.error('Failed to load annotated image');
                            e.target.style.display = 'none';
                        }}
                    />
                    <p className="text-sm text-gray-600 mt-2 text-center">
                        {segmentationResult.result.detection_count} damage segment{segmentationResult.result.detection_count !== 1 ? 's' : ''} detected
                    </p>
                </div>
            )}

            {segmentationResult && (
                <div className="bg-white border-2 border-gray-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                        Segmentation Results
                    </h3>
                    
                    {segmentationResult.result.detection_count > 0 ? (
                        <>
                            <div className="mb-4 p-3 bg-purple-50 rounded-lg">
                                <p className="text-purple-800 font-medium">
                                    Total Detections: {segmentationResult.result.detection_count}
                                </p>
                                {segmentationResult.result.has_segmentation_masks && (
                                    <p className="text-purple-700 text-sm mt-1">
                                        Segmentation masks available
                                    </p>
                                )}
                            </div>

                            <div className="space-y-4">
                                {segmentationResult.result.detections.map((detection, idx) => (
                                    <div key={idx} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                        <div className="flex justify-between items-start mb-3">
                                            <div>
                                                <p className="font-medium text-gray-800">
                                                    {detection.class_name} #{detection.id}
                                                </p>
                                                <p className="text-xs text-gray-500">
                                                    Class ID: {detection.class_id}
                                                </p>
                                            </div>
                                            <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs font-semibold rounded">
                                                {(detection.confidence * 100).toFixed(1)}% confidence
                                            </span>
                                        </div>
                                        <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                                            <p>
                                                <strong>Position:</strong> ({detection.bbox.x_min}, {detection.bbox.y_min})
                                            </p>
                                            <p>
                                                <strong>Size:</strong> {detection.bbox.width} × {detection.bbox.height}px
                                            </p>
                                            <p>
                                                <strong>BBox Area:</strong> {detection.bbox.area.toLocaleString()}px²
                                            </p>
                                            {detection.mask && (
                                                <>
                                                    <p>
                                                        <strong>Mask Area:</strong> {detection.mask.area.toLocaleString()}px²
                                                    </p>
                                                    <p className="col-span-2">
                                                        <strong>Contours:</strong> {detection.mask.contour_count}
                                                    </p>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <button
                                onClick={onNext}
                                className="mt-6 w-full bg-green-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center justify-center gap-2"
                            >
                                Proceed to Next Step
                                <ArrowRight size={20} />
                            </button>
                        </>
                    ) : (
                        <div className="text-center py-8">
                            <p className="text-gray-600 mb-4">
                                No damage segments detected in the analysis.
                            </p>
                            <button
                                onClick={onBack}
                                className="bg-purple-500 text-white py-2 px-6 rounded-lg font-medium hover:bg-purple-600 transition-colors"
                            >
                                Try Another Image
                            </button>
                        </div>
                    )}
                </div>
            )}

            <button
                onClick={onBack}
                className="w-full bg-gray-200 text-gray-800 py-2 px-4 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center justify-center gap-2"
            >
                <ArrowLeft size={20} />
                Back to Localization
            </button>
        </div>
    );
}

export default Classifier;