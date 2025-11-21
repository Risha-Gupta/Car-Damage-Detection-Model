/* eslint-disable no-unused-vars */
import { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Upload, Loader2, ArrowRight, ArrowLeft} from 'lucide-react';
import { getIsDamaged } from '../services/predectiveModels';
import { setImage, setStepStatus } from '../utils/imageSlice'; 
import { fileStore } from '../utils/fileStore';
const Predictor = ({onNext}) => {
    const dispatch = useDispatch();
    const [selectedImage, setSelectedImage] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);
    const [prediction, setPrediction] = useState(null);

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedImage(file);
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
            setResult('');
            setPrediction(null);
            fileStore.current = file; 
            dispatch(setImage({
                name: file.name,
                size: file.size,
                type: file.type,
                preview: url
            }));
        }
    };

    const handleSubmit = async () => {
        if (!selectedImage) {
            setResult('Please select an image first.');
            return;
        }

        setLoading(true);
        setResult('');
        setPrediction(null);
        dispatch(setStepStatus({ step: 1, status: "processing" }));
        try {
            const response = await getIsDamaged(selectedImage);
            const data = response.data.result;
            console.log(data)
            setPrediction(data);
            console.log(prediction)
            setResult(`Prediction complete for "${selectedImage.name}"`);
            if(data.is_damaged)
                dispatch(setStepStatus({ step: 1, status: "success" }));

            else
                dispatch(setStepStatus({ step: 1, status: "success" }));

        } catch (error) {
            console.error('Prediction error:', error);
            setResult('Failed to get prediction. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setSelectedImage(null);
        setPreviewUrl(null);
        setResult('');
        setPrediction(null);
    };
    const handleProceed = () => {
        if (prediction?.is_damaged) {
            onNext();
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 py-12 px-4">
            <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-8">
                <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
                    Damage Detection
                </h1>

                <div className="space-y-6">
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageChange}
                            className="hidden"
                            id="image-upload"
                        />
                        <label
                            htmlFor="image-upload"
                            className="cursor-pointer flex flex-col items-center"
                        >
                            <Upload className="w-16 h-16 text-gray-400 mb-4" />
                            <span className="text-lg text-gray-600">
                                Click to upload an image
                            </span>
                            <span className="text-sm text-gray-500 mt-2">
                                PNG, JPG, GIF up to 10MB
                            </span>
                        </label>
                    </div>

                    {previewUrl && (
                        <div className="mt-4">
                            <p className="text-sm font-medium text-gray-700 mb-2">Preview:</p>
                            <img
                                src={previewUrl}
                                alt="Preview"
                                className="max-w-full h-auto rounded-lg shadow-md max-h-96 mx-auto"
                            />
                            <p className="text-sm text-gray-600 mt-2 text-center">
                                {selectedImage.name}
                            </p>
                        </div>
                    )}

                    <div className="flex gap-4">
                        <button
                            onClick={handleSubmit}
                            className="flex-1 bg-blue-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center"
                            disabled={!selectedImage || loading}
                        >
                            {loading ? <Loader2 className="animate-spin mr-2" /> : null}
                            {loading ? 'Processing...' : 'Submit Image'}
                        </button>
                        <button
                            onClick={handleClear}
                            className="flex-1 bg-gray-200 text-gray-800 py-3 px-6 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                        >
                            Clear
                        </button>
                    </div>
                </div>

                {result && (
                    <div className="mt-8 p-4 bg-gray-50 border rounded-lg text-center">
                        <p className="text-gray-700">{result}</p>
                    </div>
                )}

                {prediction && (
                    <div className={`mt-6 p-6 ${
                        prediction.is_damaged
                            ? "bg-red-50 border-red-200"
                            : "bg-green-50 border-green-200"
                        } rounded-lg`}>
                        <h2 className={`text-xl font-semibold mb-3 ${
                            prediction.is_damaged ? "text-red-800" : "text-green-800"
                        }`}>Prediction Result</h2>
                        <p className={prediction.is_damaged ? "text-red-700" : "text-green-700"}>
                            <strong>Status:</strong> {prediction.status}
                        </p>
                        <p className={prediction.is_damaged ? "text-red-700" : "text-green-700"}>
                            <strong>Damage Probability:</strong> {prediction.damage_probability}
                        </p>
                        <p className={prediction.is_damaged ? "text-red-700" : "text-green-700"}>
                            <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%
                        </p>
                        {prediction.is_damaged && (
                        <button
                            onClick={handleProceed}
                            className="mt-4 w-full bg-blue-500 text-white py-2 px-4 rounded-lg font-medium hover:bg-blue-600 transition-colors flex items-center justify-center gap-2"
                        >
                            Proceed to Damage Localization
                            <ArrowRight size={20} />
                        </button>
                    )}
                    </div>
                    
                )}
            </div>
        </div>
    );
};

export default Predictor;
