import { ArrowRight, ArrowLeft, Loader2 } from "lucide-react"; 
import { useState } from "react"; 
import { getDamagedAreas } from "../services/predectiveModels"; 
import { PIPELINE_API_BASE_URL } from "../../public/constants";
import { fileStore } from "../utils/fileStore";
import { setStepStatus } from "../utils/imageSlice";
import { useDispatch } from "react-redux";

const Locator = ({ onBack, onNext }) => {
    const dispatch = useDispatch()
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
        dispatch(setStepStatus({ step: 2, status: "processing" }))
        try {
            const response = await getDamagedAreas(image);
            const data = response.data;
            setLocalizationResult(data);
            console.log(data);
            if(data.result.damage_regions_count>0)
                dispatch(setStepStatus({ step: 2, status: "success" }))
            else
                dispatch(setStepStatus({ step: 2, status: "failed" }))
        } catch (error) {
            console.error('Localization error:', error);
            setError('Failed to analyze damage areas. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 py-12 px-4">
            <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8">
                <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
                    Damage Localization
                </h1>

                <div className="space-y-6">
                    

                    {error && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                            <p className="text-red-700">{error}</p>
                        </div>
                    )}

                    {!localizationResult && (
                        <button
                            onClick={handleAnalyze}
                            className="w-full bg-blue-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-600 transition-colors disabled:opacity-50 flex justify-center items-center gap-2"
                            disabled={loading || !image}
                        >
                            {loading ? <Loader2 className="animate-spin" /> : null}
                            {loading ? 'Analyzing...' : 'Run Localization Analysis'}
                        </button>
                    )}

                    {localizationResult && (
                        <>
                            {localizationResult.result.annotated_image_path && (
                                <div className="bg-white border border-gray-200 rounded-lg p-6">
                                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                                        Annotated Image
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
                                    <p className="text-center mt-4 text-gray-600 font-medium">
                                        {localizationResult.result.damage_regions_count} damage area{localizationResult.result.damage_regions_count !== 1 ? 's' : ''} detected
                                    </p>
                                </div>
                            )}

                            <div className="bg-white border border-gray-200 rounded-lg p-6">
                                <h3 className="text-lg font-semibold text-gray-800 mb-4">
                                    Detection Details
                                </h3>
                                
                                {localizationResult.result.damage_regions_count > 0 ? (
                                    <>
                                        <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                                            <p className="text-blue-800 font-medium text-center">
                                                Total Detections: {localizationResult.result.damage_regions_count}
                                            </p>
                                        </div>
                                        <div className="space-y-4">
                                            {localizationResult.result.detected_damage_bboxes.map((bbox, idx) => (
                                                <div key={idx} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                                    <div className="flex justify-between items-center mb-3">
                                                        <p className="font-medium text-gray-800">
                                                            Damage Area #{idx + 1}
                                                        </p>
                                                        <span className="px-3 py-1 bg-blue-100 text-blue-800 text-sm font-semibold rounded-full">
                                                            {(bbox.confidence * 100).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                    <div className="grid grid-cols-2 gap-3 text-sm text-gray-600">
                                                        <div>
                                                            <p className="font-medium text-gray-700">Position</p>
                                                            <p>({bbox.x_min}, {bbox.y_min})</p>
                                                        </div>
                                                        <div>
                                                            <p className="font-medium text-gray-700">Size</p>
                                                            <p>{bbox.x_max - bbox.x_min} × {bbox.y_max - bbox.y_min}px</p>
                                                        </div>
                                                        <div className="col-span-2">
                                                            <p className="font-medium text-gray-700">Area</p>
                                                            <p>{bbox.bbox_area.toLocaleString()}px²</p>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                ) : (
                                    <div className="text-center py-8">
                                        <p className="text-gray-600 text-lg">
                                            No damage areas detected in the localization analysis.
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
                            Back to Detection
                        </button>
                        {localizationResult && (
                            <button
                                onClick={onNext}
                                className="flex-1 bg-green-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center justify-center gap-2"
                            >
                                Proceed to Classification
                                <ArrowRight size={20} />
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Locator;