import { ArrowRight, ArrowLeft, Loader2 } from "lucide-react"; 
import { useState } from "react"; 
import { getDamageTypes } from "../services/predectiveModels"; 
import { PIPELINE_API_BASE_URL } from "../../public/constants";
import { fileStore } from "../utils/fileStore";
import { useDispatch } from "react-redux";
import { setStepStatus } from "../utils/imageSlice";

const Classifier = ({ onBack, onNext }) => {
    const dispatch = useDispatch()
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
        dispatch(setStepStatus({ step: 3, status: "processing" }))
        try {
            const response = await getDamageTypes(image);
            const data = response.data;
            setSegmentationResult(data);
            if(data.result.detection_count>0)
                dispatch(setStepStatus({ step: 3, status: "success" }))
            else 
                dispatch(setStepStatus({ step: 3, status: "failed" }))
            console.log(data);
        } catch (error) {
            console.error('Segmentation error:', error);
            setError('Failed to analyze damage segmentation. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 py-12 px-4">
            <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8">
                <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
                    Damage Classification
                </h1>

                <div className="space-y-6">
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-6">
                        <h2 className="text-xl font-semibold text-purple-800 mb-2">
                            Analyze Damage Types
                        </h2>
                        <p className="text-purple-700">
                            Using YOLOv9 segmentation model to classify and segment damaged areas
                        </p>
                    </div>

                    {error && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                            <p className="text-red-700">{error}</p>
                        </div>
                    )}

                    {!segmentationResult && (
                        <button
                            onClick={handleAnalyze}
                            className="w-full bg-purple-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-purple-600 transition-colors disabled:opacity-50 flex justify-center items-center gap-2"
                            disabled={loading || !image}
                        >
                            {loading ? <Loader2 className="animate-spin" /> : null}
                            {loading ? 'Analyzing...' : 'Run Segmentation Analysis'}
                        </button>
                    )}

                    {segmentationResult && (
                        <>
                            {segmentationResult.result?.annotated_image && (
                                <div className="bg-white border border-gray-200 rounded-lg p-6">
                                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                                        Segmented Image
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
                                    <p className="text-center mt-4 text-gray-600 font-medium">
                                        {segmentationResult.result.detection_count} damage segment{segmentationResult.result.detection_count !== 1 ? 's' : ''} detected
                                    </p>
                                </div>
                            )}

                            <div className="bg-white border border-gray-200 rounded-lg p-6">
                                <h3 className="text-lg font-semibold text-gray-800 mb-4">
                                    Segmentation Details
                                </h3>
                                
                                {segmentationResult.result.detection_count > 0 ? (
                                    <>
                                        <div className="mb-6 p-4 bg-purple-50 rounded-lg">
                                            <p className="text-purple-800 font-medium text-center">
                                                Total Detections: {segmentationResult.result.detection_count}
                                            </p>
                                            {segmentationResult.result.has_segmentation_masks && (
                                                <p className="text-purple-700 text-sm mt-1 text-center">
                                                    Segmentation masks available
                                                </p>
                                            )}
                                        </div>

                                        <div className="space-y-4">
                                            {segmentationResult.result.detections.map((detection, idx) => (
                                                <div key={idx} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                                    <div className="flex justify-between items-center mb-3">
                                                        <div>
                                                            <p className="font-medium text-gray-800 text-lg">
                                                                {detection.class_name.toUpperCase()}
                                                            </p>
                                                        </div>
                                                        <span className="px-3 py-1 bg-purple-100 text-purple-800 text-sm font-semibold rounded-full">
                                                            {(detection.confidence * 100).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                    <div className="grid grid-cols-2 gap-3 text-sm text-gray-600">
                                                        <div>
                                                            <p className="font-medium text-gray-700">Position</p>
                                                            <p>({detection.bbox.x_min}, {detection.bbox.y_min})</p>
                                                        </div>
                                                        <div>
                                                            <p className="font-medium text-gray-700">Size</p>
                                                            <p>{detection.bbox.width} × {detection.bbox.height}px</p>
                                                        </div>
                                                        <div>
                                                            <p className="font-medium text-gray-700">BBox Area</p>
                                                            <p>{detection.bbox.area.toLocaleString()}px²</p>
                                                        </div>
                                                        {detection.mask && (
                                                            <>
                                                                <div>
                                                                    <p className="font-medium text-gray-700">Mask Area</p>
                                                                    <p>{detection.mask.area.toLocaleString()}px²</p>
                                                                </div>
                                                                
                                                            </>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                ) : (
                                    <div className="text-center py-8">
                                        <p className="text-gray-600 text-lg">
                                            No damage segments detected in the analysis.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </>
                    )}

                    <div className="flex gap-4 pt-4">
                        <button
                            onClick={onBack}
                            className="flex-1 bg-gray-200 text-gray-800 py-3 px-6 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center justify-center gap-2"
                        >
                            <ArrowLeft size={20} />
                            Back to Localization
                        </button>
                        {segmentationResult && (
                            <button
                                onClick={onNext}
                                className="flex-1 bg-green-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center justify-center gap-2"
                            >
                                Proceed to Next Step
                                <ArrowRight size={20} />
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Classifier;