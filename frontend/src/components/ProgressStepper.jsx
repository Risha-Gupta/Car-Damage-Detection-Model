import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { setStep } from '../utils/imageSlice'; 

const ProgressStepper = () => {

    const dispatch = useDispatch();

    const currentStep = useSelector((state) => state.image.step);
    const stepStatus = useSelector((state) => state.image.stepStatus);

    const steps = [
        { number: 1, label: 'Stage 1' },
        { number: 2, label: 'Stage 2' },
        { number: 3, label: 'Stage 3' },
        { number: 4, label: 'Stage 4' },
        { number: 5, label: 'Stage 5' }
    ];

    const handleStepClick = (stepNumber) => {
        dispatch(setStep(stepNumber));
    };

    return (
        <div className="w-48 shrink-0 min-h-screen bg-gray-50 p-6 border-r border-gray-200">
            <div className="flex flex-col items-center space-y-2">
                {steps.map((step, index) => {

                    const status = stepStatus[step.number];

                    const isFailed = status === "failed";
                    const isDone = status === "success";
                    const isActive = currentStep === step.number;
                    const isProcessing = status === "processing";

                    let circleStyle = "";
                    let displaySymbol = step.number;

                    if (isFailed) {
                        displaySymbol = "✗";
                        circleStyle = "bg-red-500 text-white hover:bg-red-600";
                    }
                    else if (isDone) {
                        displaySymbol = "✓";
                        circleStyle = "bg-green-500 text-white hover:bg-green-600";
                    }
                    else if (isProcessing) {
                        displaySymbol = step.number;
                        circleStyle = "bg-yellow-400 text-black animate-pulse";
                    }
                    else if (isActive) {
                        circleStyle = "bg-blue-500 text-white hover:bg-blue-600";
                    }
                    else {
                        circleStyle = "bg-gray-300 text-gray-600 hover:bg-gray-400";
                    }

                    return (
                        <React.Fragment key={step.number}>
                            <div className="flex flex-col items-center">
                                <div
                                    onClick={() => handleStepClick(step.number)}
                                    className={`w-12 h-12 rounded-full flex items-center justify-center 
                                        font-semibold text-lg transition-all duration-300 cursor-pointer
                                        ${circleStyle} hover:scale-110 active:scale-95`}
                                >
                                    {displaySymbol}
                                </div>

                                <span
                                    onClick={() => handleStepClick(step.number)}
                                    className={`mt-2 text-sm font-medium cursor-pointer transition-all duration-200 
                                        ${
                                            isFailed
                                                ? "text-red-600"
                                                : isDone
                                                    ? "text-green-700"
                                                    : isActive
                                                        ? "text-blue-700"
                                                        : "text-gray-500"
                                        }`}
                                >
                                    {step.label}
                                </span>
                            </div>

                            {/* Vertical connector line */}
                            {index < steps.length - 1 && (
                                <div className="w-1 h-8">
                                    <div
                                        className={`w-full h-full transition-all duration-300 rounded ${
                                            isFailed
                                                ? "bg-red-500"
                                                : isDone
                                                    ? "bg-green-500"
                                                    : "bg-gray-300"
                                        }`}
                                    />
                                </div>
                            )}
                        </React.Fragment>
                    );
                })}
            </div>
        </div>
    );
};

export default ProgressStepper;