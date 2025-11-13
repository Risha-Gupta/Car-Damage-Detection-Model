import { ArrowRight, ArrowLeft, Loader2 } from "lucide-react"; 
import { useState } from "react"; 
import { getDamagedAreas } from "../services/predectiveModels"; 
import { PIPELINE_API_BASE_URL } from "../../public/constants";
import { fileStore } from "../utils/fileStore";

const Locator = ({ onBack, onNext }) => {
    const [loading, setLoading] = useState(false);
    const [localizationResult, setLocalizationResult] = useState(null);
    const [error, setError] = useState(null);
    const image = fileStore.current;
    
    const handleAnalyze = async () => {
        if (!image) {
            setError('No image available. Please go back and upload an image.');
            return;
        }

        setLoading(true);
        setError(null);
        setLocalizationResult(null);

        try {
            const response = await getDamagedAreas(image);
            const data = response.data;
            setLocalizationResult(data);
            console.log(data);
        } catch (error) {
            console.error('Localization error:', error);
            setError('Failed to analyze damage areas. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-blue-800 mb-3">
                    Damage Localization Analysis
                </h2>
                <p className="text-blue-700">
                    Analyzing damaged areas using YOLOv8 detection model...
                </p>
            </div>

            {error && (
                <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4">
                    <p className="text-red-700">{error}</p>
                </div>
            )}

            <button
                onClick={handleAnalyze}
                className="w-full bg-blue-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-600 transition-colors disabled:opacity-50 flex justify-center items-center gap-2"
                disabled={loading || !image}
            >
                {loading ? <Loader2 className="animate-spin" /> : null}
                {loading ? 'Analyzing...' : 'Run Localization Analysis'}
            </button>

            {localizationResult?.result.annotated_image_path && (
                <div className="bg-white border-2 border-gray-200 rounded-lg p-4">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">
                        Detected Damage Localization
                    </h3>
                    <img 
                        src={`${PIPELINE_API_BASE_URL}${localizationResult.result.annotated_image_path}`}
                        alt="Annotated damage localization"
                        className="w-full h-auto rounded-lg shadow-md"
                        onError={(e) => {
                            console.error('Failed to load annotated image');
                            e.target.style.display = 'none';
                        }}
                    />
                    <p className="text-sm text-gray-600 mt-2 text-center">
                        {localizationResult.result.damage_regions_count} damage area{localizationResult.result.damage_regions_count !== 1 ? 's' : ''} detected
                    </p>
                </div>
            )}

            {localizationResult && (
                <div className="bg-white border-2 border-gray-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                        Localization Results
                    </h3>
                    
                    {localizationResult.result.damage_regions_count > 0 ? (
                        <>
                            <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                                <p className="text-blue-800 font-medium">
                                    Total Detections: {localizationResult.result.damage_regions_count}
                                </p>
                            </div>

                            <div className="space-y-4">
                                {localizationResult.result.detected_damage_bboxes.map((bbox, idx) => (
                                    <div key={idx} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                        <div className="flex justify-between items-start mb-3">
                                            <p className="font-medium text-gray-800">
                                                Damage Area #{idx + 1}
                                            </p>
                                            <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-semibold rounded">
                                                {(bbox.confidence * 100).toFixed(1)}% confidence
                                            </span>
                                        </div>
                                        <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                                            <p>
                                                <strong>Position:</strong> ({bbox.x_min}, {bbox.y_min})
                                            </p>
                                            <p>
                                                <strong>Size:</strong> {bbox.x_max - bbox.x_min} × {bbox.y_max - bbox.y_min}px
                                            </p>
                                            <p>
                                                <strong>Area:</strong> {bbox.bbox_area}px²
                                            </p>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            <button
                                onClick={onNext}
                                className="mt-6 w-full bg-green-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center justify-center gap-2"
                            >
                                Proceed to Classification
                                <ArrowRight size={20} />
                            </button>
                        </>
                    ) : (
                        <div className="text-center py-8">
                            <p className="text-gray-600 mb-4">
                                No damage areas detected in the localization analysis.
                            </p>
                            <button
                                onClick={onBack}
                                className="bg-blue-500 text-white py-2 px-6 rounded-lg font-medium hover:bg-blue-600 transition-colors"
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
                Back to Detection
            </button>
        </div>
    );
}

export default Locator;