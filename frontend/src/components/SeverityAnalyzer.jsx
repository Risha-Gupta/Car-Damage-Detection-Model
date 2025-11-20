import { ArrowRight, ArrowLeft, Loader2, AlertCircle, CheckCircle } from "lucide-react"; 
import { useState } from "react"; 
import { fileStore } from "../utils/fileStore";
import { classifyDamage } from "../services/predectiveModels";

const SeverityAnalyzer = ({ onBack, onNext }) => {
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

    const getDamageTypeIcon = (type) => {
        switch(type) {
            case 'dent': return '🔨';
            case 'crack': return '⚡';
            case 'scratch': return '✏️';
            case 'no_damage': return '✅';
            default: return '❓';
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

        try {
            const response = await classifyDamage(image)
            const data = response.data
            setClassificationResult(data);
            console.log('Classification result:', data);
        } catch (error) {
            console.error('Classification error:', error);
            setError('Failed to classify damage. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="bg-indigo-50 border-2 border-indigo-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-indigo-800 mb-3">
                    🎯 Stage 4: Damage Classification & Severity Analysis
                </h2>
                <p className="text-indigo-700">
                    Analyzing damage severity, type, coverage, and repair priority...
                </p>
            </div>

            {error && (
                <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4 flex items-start gap-3">
                    <AlertCircle className="text-red-600 shrink-0" size={24} />
                    <p className="text-red-700">{error}</p>
                </div>
            )}

            <button
                onClick={handleClassify}
                className="w-full bg-indigo-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-indigo-600 transition-colors disabled:opacity-50 flex justify-center items-center gap-2"
                disabled={loading || !image}
            >
                {loading ? <Loader2 className="animate-spin" /> : null}
                {loading ? 'Classifying Damage...' : 'Run Damage Classification'}
            </button>

            {classificationResult && (
                <div className="space-y-6">
                    {/* Main Classification Card */}
                    <div className="bg-white border-2 border-gray-200 rounded-lg p-6 shadow-lg">
                        <div className="flex items-center gap-2 mb-6">
                            <CheckCircle className="text-green-600" size={24} />
                            <h3 className="text-xl font-semibold text-gray-800">
                                Classification Complete
                            </h3>
                        </div>
                        
                        {/* Severity Badge */}
                        <div className={`mb-6 p-4 rounded-lg border-2 ${getSeverityColor(classificationResult.result.damage_severity)}`}>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm font-medium opacity-75 mb-1">Damage Severity</p>
                                    <p className="text-2xl font-bold capitalize">
                                        {classificationResult.result.damage_severity}
                                    </p>
                                </div>
                                <div className="text-5xl">
                                    {classificationResult.result.damage_severity === 'minor' ? '✓' : 
                                     classificationResult.result.damage_severity === 'moderate' ? '⚠️' : '🚨'}
                                </div>
                            </div>
                        </div>

                        {/* Grid of Metrics */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Damage Type */}
                            <div className="bg-linear-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
                                <p className="text-sm text-purple-700 font-medium mb-2">Damage Type</p>
                                <div className="flex items-center gap-2">
                                    <span className="text-3xl">{getDamageTypeIcon(classificationResult.result.damage_type)}</span>
                                    <p className="text-xl font-bold text-purple-900 capitalize">
                                        {classificationResult.result.damage_type.replace('_', ' ')}
                                    </p>
                                </div>
                            </div>

                            {/* Repair Priority */}
                            <div className={`p-4 rounded-lg border ${getPriorityColor(classificationResult.result.repair_priority)}`}>
                                <p className="text-sm font-medium opacity-75 mb-2">Repair Priority</p>
                                <p className="text-xl font-bold capitalize">
                                    {classificationResult.result.repair_priority}
                                </p>
                            </div>

                            {/* Coverage Percentage */}
                            <div className="bg-linear-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
                                <p className="text-sm text-blue-700 font-medium mb-2">Damage Coverage</p>
                                <div className="flex items-baseline gap-2">
                                    <p className="text-3xl font-bold text-blue-900">
                                        {classificationResult.result.damage_coverage_percent}
                                    </p>
                                    <span className="text-xl text-blue-700">%</span>
                                </div>
                                <div className="mt-2 bg-blue-200 rounded-full h-2">
                                    <div 
                                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${Math.min(classificationResult.result.damage_coverage_percent, 100)}%` }}
                                    />
                                </div>
                            </div>

                            {/* Detection Count */}
                            <div className="bg-linear-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
                                <p className="text-sm text-green-700 font-medium mb-2">Regions Detected</p>
                                <div className="flex items-baseline gap-2">
                                    <p className="text-3xl font-bold text-green-900">
                                        {classificationResult.result.detection_count}
                                    </p>
                                    <span className="text-xl text-green-700">
                                        region{classificationResult.result.detection_count !== 1 ? 's' : ''}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                    {classificationResult.result.roi_images && classificationResult.result.roi_images.length > 0 && (
                        <div className="bg-white border-2 border-indigo-200 rounded-lg p-6">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                                🔍 Zoomed Damage Regions ({classificationResult.result.roi_count})
                            </h3>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {classificationResult.result.roi_images.map((roi, idx) => (
                                    <div key={idx} className="bg-gray-50 border border-gray-300 rounded-lg overflow-hidden hover:shadow-lg transition-shadow">
                                        <div className="relative">
                                            <img 
                                                src={`data:image/jpeg;base64,${roi.base64}`}
                                                alt={`ROI ${roi.roi_id}`}
                                                className="w-full h-48 object-cover"
                                            />
                                            <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-xs font-semibold">
                                                ROI #{roi.roi_id}
                                            </div>
                                        </div>
                                        
                                        <div className="p-3 space-y-2">
                                            <div className="flex justify-between items-center">
                                                <span className="text-sm font-medium text-gray-700">
                                                    {roi.class_name}
                                                </span>
                                                {roi.confidence && (
                                                    <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded font-semibold">
                                                        {(roi.confidence * 100).toFixed(1)}%
                                                    </span>
                                                )}
                                            </div>
                                            
                                            <div className="text-xs text-gray-600 space-y-1">
                                                <p>
                                                    <strong>Dimensions:</strong> {roi.roi_dimensions.width} × {roi.roi_dimensions.height}px
                                                </p>
                                                <p>
                                                    <strong>Area:</strong> {roi.bbox_area.toLocaleString()}px²
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Action Recommendations */}
                    <div className="bg-linear-to-r from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-lg p-6">
                        <h4 className="font-semibold text-indigo-900 mb-3 flex items-center gap-2">
                            💡 Recommended Actions
                        </h4>
                        <ul className="space-y-2 text-indigo-800">
                            {classificationResult.result.repair_priority === 'high' && (
                                <li className="flex items-start gap-2">
                                    <span className="text-red-600 font-bold">•</span>
                                    <span>Immediate repair required - Schedule assessment ASAP</span>
                                </li>
                            )}
                            {classificationResult.result.repair_priority === 'medium' && (
                                <li className="flex items-start gap-2">
                                    <span className="text-orange-600 font-bold">•</span>
                                    <span>Schedule repair within 2-4 weeks</span>
                                </li>
                            )}
                            {classificationResult.result.repair_priority === 'low' && (
                                <li className="flex items-start gap-2">
                                    <span className="text-blue-600 font-bold">•</span>
                                    <span>Minor repair - can be addressed during routine maintenance</span>
                                </li>
                            )}
                            {classificationResult.result.damage_coverage_percent > 10 && (
                                <li className="flex items-start gap-2">
                                    <span className="text-purple-600 font-bold">•</span>
                                    <span>Large affected area - consider professional inspection</span>
                                </li>
                            )}
                            <li className="flex items-start gap-2">
                                <span className="text-green-600 font-bold">•</span>
                                <span>Document all findings for insurance or maintenance records</span>
                            </li>
                        </ul>
                    </div>

                    {/* Navigation Buttons */}
                    <div className="flex gap-4">
                        <button
                            onClick={onNext}
                            className="flex-1 bg-green-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center justify-center gap-2"
                        >
                            Complete Analysis
                            <ArrowRight size={20} />
                        </button>
                    </div>
                </div>
            )}

            <button
                onClick={onBack}
                className="w-full bg-gray-200 text-gray-800 py-2 px-4 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center justify-center gap-2"
            >
                <ArrowLeft size={20} />
                Back to Segmentation
            </button>
        </div>
    );
}

export default SeverityAnalyzer;