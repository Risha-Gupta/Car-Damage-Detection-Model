import { ArrowRight, ArrowLeft, Loader2, DollarSign, Car, AlertTriangle, CheckCircle, IndianRupee } from "lucide-react";
import { useState } from "react";
import { estimateCost } from "../services/predectiveModels";
import { fileStore } from "../utils/fileStore";
import { useDispatch } from "react-redux";
import { setStepStatus } from "../utils/imageSlice";
import DownloadReportButton from "./DownloadReportButton";
const Estimator = ({ onBack, onNext }) => {
    const dispatch = useDispatch();
    const [loading, setLoading] = useState(false);
    const [estimationResult, setEstimationResult] = useState(null);
    const [error, setError] = useState(null);
    const image = fileStore.current;

    const handleEstimate = async () => {
        if (!image) {
            setError('No image available. Please go back and upload an image.');
            return;
        }

        setLoading(true);
        setError(null);
        setEstimationResult(null);
        dispatch(setStepStatus({ step: 5, status: "processing" }));

        try {
            const response = await estimateCost(image);
            const data = response.data;
            setEstimationResult(data);
            console.log(data)
            if (data.result.valid_regions > 0) {
                dispatch(setStepStatus({ step: 5, status: "success" }));
            } else {
                dispatch(setStepStatus({ step: 5, status: "failed" }));
            }
        } catch (error) {
            console.error('Cost estimation error:', error);
            setError('Failed to estimate repair costs. Please try again.');
            dispatch(setStepStatus({ step: 5, status: "failed" }));
        } finally {
            setLoading(false);
        }
    };

    const formatCurrency = (amount) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'INR'
        }).format(amount);
    };

    const getSeverityColor = (severity) => {
        switch (severity?.toLowerCase()) {
            case 'minor': return 'bg-green-100 text-green-800';
            case 'moderate': return 'bg-yellow-100 text-yellow-800';
            case 'severe': return 'bg-red-100 text-red-800';
            default: return 'bg-gray-100 text-gray-800';
        }
    };

    return (
        <div className="space-y-6">
            <div className="bg-emerald-50 border-2 border-emerald-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-emerald-800 mb-3">
                    Repair Cost Estimation
                </h2>
                <p className="text-emerald-700">
                    Estimating repair costs...
                </p>
            </div>

            {error && (
                <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4">
                    <p className="text-red-700">{error}</p>
                </div>
            )}

            <button
                onClick={handleEstimate}
                className="w-full bg-emerald-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-emerald-600 transition-colors disabled:opacity-50 flex justify-center items-center gap-2"
                disabled={loading || !image}
            >
                {loading ? <Loader2 className="animate-spin" /> : <IndianRupee size={20} />}
                {loading ? 'Estimating Costs...' : 'Run Cost Estimation'}
            </button>

            {estimationResult && (
                <div className="space-y-4">
                    <div className="bg-white border-2 border-gray-200 rounded-lg p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold text-gray-800">
                                Estimation Summary
                            </h3>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div className="bg-emerald-50 p-4 rounded-lg text-center">
                                <p className="text-sm text-emerald-600 mb-1">Total Estimated Cost</p>
                                <p className="text-2xl font-bold text-emerald-800">
                                    {formatCurrency(estimationResult.result.total_estimated_cost)}
                                </p>
                            </div>
                            <div className="bg-blue-50 p-4 rounded-lg text-center">
                                <p className="text-sm text-blue-600 mb-1">Valid Regions</p>
                                <p className="text-2xl font-bold text-blue-800">
                                    {estimationResult.result.valid_regions}
                                </p>
                            </div>
                        </div>

                        {estimationResult.result.ignored_regions > 0 && (
                            <div className="flex items-center gap-2 p-3 bg-amber-50 rounded-lg mb-4">
                                <AlertTriangle size={18} className="text-amber-600" />
                                <p className="text-sm text-amber-700">
                                    {estimationResult.result.ignored_regions} region(s) ignored due to unknown part classification
                                </p>
                            </div>
                        )}

                        {estimationResult.result.note && (
                            <div className="p-3 bg-gray-50 rounded-lg">
                                <p className="text-sm text-gray-600">{estimationResult.result.note}</p>
                            </div>
                        )}
                    </div>

                    {estimationResult.result.details && estimationResult.result.details.length > 0 && (
                        <div className="bg-white border-2 border-gray-200 rounded-lg p-6">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">
                                Cost Breakdown by Part
                            </h3>
                            <div className="space-y-4">
                                {estimationResult.result.details.map((detail, idx) => (
                                    <div key={idx} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                        <div className="flex justify-between items-start mb-3">
                                            <div className="flex items-center gap-2">
                                                <Car size={18} className="text-gray-600" />
                                                <p className="font-medium text-gray-800 capitalize">
                                                    {detail.part.replace(/_/g, ' ')}
                                                </p>
                                            </div>
                                            <span className={`px-2 py-1 text-xs font-semibold rounded ${getSeverityColor(detail.severity)}`}>
                                                {detail.severity}
                                            </span>
                                        </div>
                                        <div className="grid grid-cols-2 gap-3 text-sm mb-3">
                                            <p className="text-gray-600">
                                                <span className="font-medium">Damage Type:</span> {detail.damage_type}
                                            </p>
                                            <p className="text-gray-600">
                                                <span className="font-medium">Coverage:</span> {detail.coverage_percent.toFixed(1)}%
                                            </p>
                                        </div>
                                        <div className="flex justify-between items-center pt-3 border-t border-gray-200">
                                            <span className="text-sm text-gray-600">Final Estimate</span>
                                            <span className="text-lg font-bold text-emerald-700">
                                                {formatCurrency(detail.final_cost)}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    <DownloadReportButton 
                        downloadUrl={estimationResult.report.download_url}
                    />
                </div>
            )}
            <button
                onClick={onBack}
                className="w-full bg-gray-200 text-gray-800 py-2 px-4 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center justify-center gap-2"
            >
                <ArrowLeft size={20} />
                Back to Classification
            </button>
        </div>
    );
};

export default Estimator;