import { ArrowRight, ArrowLeft, Loader2, AlertCircle, CheckCircle } from "lucide-react"; 
import { useState } from "react"; 
import { fileStore } from "../utils/fileStore";
import { classifyDamage } from "../services/predectiveModels";
import { useDispatch } from "react-redux";
import { setStepStatus } from "../utils/imageSlice";

const SeverityAnalyzer = ({ onBack, onNext }) => {
    const dispatch = useDispatch()
    const [loading, setLoading] = useState(false);
    const [classificationResult, setClassificationResult] = useState(null);
    const [error, setError] = useState(null);
    const image = fileStore.current;
    
    const getSeverityColor = (severity) => {
        switch(severity) {
            case 'minor': return 'text-green-700 bg-green-100 border-green-300';
            case 'moderate': return 'text-yellow-700 bg-yellow-100 border-yellow-300';
            case 'severe': return 'text-red-700 bg-red-100 border-red-300';
            default: return 'text-gray-700 bg-gray-100 border-gray-300';
        }
    };

    const getPriorityColor = (priority) => {
        switch(priority) {
            case 'low': return 'text-blue-700 bg-blue-100';
            case 'medium': return 'text-orange-700 bg-orange-100';
            case 'high': return 'text-red-700 bg-red-100';
            default: return 'text-gray-700 bg-gray-100';
        }
    };

    const handleClassify = async () => {
        if (!image) {
            setError('No image available. Please go back and upload an image.');
            return;
        }

        setLoading(true);
        setError(null);
        setClassificationResult(null);
        dispatch(setStepStatus({ step: 4, status: "processing" }))
        try {
            const res = await classifyDamage(image)
            const data = res.data;
            setClassificationResult(data);
            dispatch(setStepStatus({ step: 4, status: "success" }))
            console.log("Updated Stage 4:", data)
        } catch (error) {
            console.error('Classification error:', error);
            setError('Failed to classify damage. Please try again.');
            dispatch(setStepStatus({ step: 2, status: "failed" }))
        } finally {
            setLoading(false);
        }
    };

    if (!classificationResult) {
        return (
            <div className="min-h-screen bg-gray-100 py-12 px-4">
                <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8">
                    <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
                        Severity Analysis
                    </h1>

                    <div className="space-y-6">
                        <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-6">
                            <h2 className="text-xl font-semibold text-indigo-800 mb-2">
                                Damage Classification & Severity Analysis
                            </h2>
                            <p className="text-indigo-700">
                                Analyzing damage severity, type, coverage, repair priority, and the exact part of the car affected
                            </p>
                        </div>

                        {error && (
                            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                                <AlertCircle className="text-red-600 shrink-0" size={24} />
                                <p className="text-red-700">{error}</p>
                            </div>
                        )}

                        <button
                            onClick={handleClassify}
                            className="w-full bg-indigo-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-indigo-600 transition-colors disabled:opacity-50 flex justify-center items-center gap-2"
                            disabled={loading || !image}
                        >
                            {loading && <Loader2 className="animate-spin" />}
                            {loading ? 'Classifying Damage...' : 'Run Damage Classification'}
                        </button>

                        <button
                            onClick={onBack}
                            className="w-full bg-gray-200 text-gray-800 py-3 px-6 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center justify-center gap-2"
                        >
                            <ArrowLeft size={20} />
                            Back to Segmentation
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    const result = classificationResult.result;

    return (
        <div className="min-h-screen bg-gray-100 py-12 px-4">
            <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8">
                <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
                    Severity Analysis Results
                </h1>

                <div className="space-y-6">
                    <div className="bg-white border border-gray-200 rounded-lg p-6">
                        <div className="flex items-center gap-2 mb-6">
                            <CheckCircle className="text-green-600" size={24} />
                            <h3 className="text-xl font-semibold text-gray-800">
                                Classification Complete
                            </h3>
                        </div>

                        <div className={`mb-6 p-4 rounded-lg border-2 ${getSeverityColor(result.severity)}`}>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm font-medium opacity-75 mb-1">Damage Severity</p>
                                    <p className="text-2xl font-bold capitalize">
                                        {result.severity}
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                                <p className="text-sm text-purple-700 font-medium mb-2">Damage Type</p>
                                <div className="flex items-center gap-2">
                                    <p className="text-xl font-bold text-purple-900 capitalize">
                                        {result.damage_type.replace('_', ' ')}
                                    </p>
                                </div>
                            </div>

                            <div className={`p-4 rounded-lg border ${getPriorityColor(result.repair_priority)}`}>
                                <p className="text-sm font-medium opacity-75 mb-2">Repair Priority</p>
                                <p className="text-xl font-bold capitalize">
                                    {result.repair_priority}
                                </p>
                            </div>

                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                <p className="text-sm text-blue-700 font-medium mb-2">Damage Coverage</p>
                                <div className="flex items-baseline gap-2">
                                    <p className="text-3xl font-bold text-blue-900">
                                        {result.coverage_percent}
                                    </p>
                                    <span className="text-xl text-blue-700">%</span>
                                </div>
                                <div className="mt-2 bg-blue-200 rounded-full h-2">
                                    <div 
                                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${Math.min(result.coverage_percent, 100)}%` }}
                                    />
                                </div>
                            </div>

                            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                <p className="text-sm text-green-700 font-medium mb-2">Damage Regions</p>
                                <div className="flex items-baseline gap-2">
                                    <p className="text-3xl font-bold text-green-900">
                                        {result.merged_results.length}
                                    </p>
                                    <span className="text-xl text-green-700">
                                        region{result.merged_results.length !== 1 ? 's' : ''}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {result.roi_images && result.roi_images.length > 0 && (
                        <div className="bg-white border border-indigo-200 rounded-lg p-6">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">
                                Zoomed Damage Regions ({result.roi_images.length})
                            </h3>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {result.roi_images.map((roi, idx) => (
                                    <div key={idx} className="bg-gray-50 border border-gray-300 rounded-lg overflow-hidden hover:shadow-lg transition-shadow">
                                        <div className="relative">
                                            <img 
                                                src={`data:image/jpeg;base64,${roi.base64}`}
                                                alt={`ROI ${roi.roi_id}`}
                                                className="w-full h-48 object-cover"
                                            />
                                        </div>
                                        
                                        <div className="p-3 space-y-2">
                                            <div className="flex justify-between items-center">
                                                <span className="text-sm font-medium text-gray-700 capitalize">
                                                    {(roi.part_detected?.class_name) || "Unknown part"}
                                                </span>
                                                {roi.part_detected?.confidence && (
                                                    <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded font-semibold">
                                                        {(roi.part_detected.confidence * 100).toFixed(1)}%
                                                    </span>
                                                )}
                                            </div>

                                            <div className="text-xs text-gray-600 space-y-1">
                                                <p><strong>BBox:</strong> {roi.damage_bbox?.join(", ")}</p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-6">
                        <h4 className="font-semibold text-indigo-900 mb-3">
                            Recommended Actions
                        </h4>
                        <ul className="space-y-2 text-indigo-800">
                            {result.repair_priority === 'high' &&
                                <li><span className="text-red-600 font-bold">•</span> Immediate repair recommended</li>
                            }
                            {result.repair_priority === 'medium' &&
                                <li><span className="text-orange-600 font-bold">•</span> Schedule repair within 2–4 weeks</li>
                            }
                            {result.repair_priority === 'low' &&
                                <li><span className="text-blue-600 font-bold">•</span> Can be handled during routine maintenance</li>
                            }
                            <li><span className="text-green-600 font-bold">•</span> Document the damage for insurance</li>
                        </ul>
                    </div>

                    <div className="flex gap-4 pt-4">
                        <button
                            onClick={onBack}
                            className="flex-1 bg-gray-200 text-gray-800 py-3 px-6 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center justify-center gap-2"
                        >
                            <ArrowLeft size={20} />
                            Back to Segmentation
                        </button>
                        <button
                            onClick={onNext}
                            className="flex-1 bg-green-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center justify-center gap-2"
                        >
                            Complete Analysis
                            <ArrowRight size={20} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SeverityAnalyzer;