import React, { useState } from 'react';
import { Upload } from 'lucide-react';

const Body = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [result, setResult] = useState('');

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedImage(file);
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
            setResult('');
        }
    };

    const handleSubmit = () => {
        if (selectedImage) {
            setResult(`Image "${selectedImage.name}" uploaded successfully! Size: ${(selectedImage.size / 1024).toFixed(2)} KB`);
        } else {
            setResult('Please select an image first.');
        }
    };

    const handleClear = () => {
        setSelectedImage(null);
        setPreviewUrl(null);
        setResult('');
    };

    return (
        <div className="min-h-screen bg-gray-100 py-12 px-4">
            <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-8">
                <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
                    Image Upload
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
                            className="flex-1 bg-blue-500 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={!selectedImage}
                        >
                            Submit Image
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
                    <div className="mt-8 p-6 bg-green-50 border border-green-200 rounded-lg">
                        <h2 className="text-xl font-semibold text-green-800 mb-2">Result:</h2>
                        <p className="text-green-700">{result}</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Body;