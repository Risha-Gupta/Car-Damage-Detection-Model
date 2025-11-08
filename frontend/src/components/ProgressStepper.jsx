import React from 'react';
import { useDispatch } from 'react-redux';
import { setStep } from '../utils/imageSlice'; 
const ProgressStepper = ({ currentStep, setCurrentStep }) => {
    const dispatch = useDispatch();
    
    const steps = [
        { number: 1, label: 'Stage 1' },
        { number: 2, label: 'Stage 2' },
        { number: 3, label: 'Stage 3' },
        { number: 4, label: 'Stage 4' },
        { number: 5, label: 'Stage 5' }
    ];

    const handleStepClick = (stepNumber) => {
        setCurrentStep(stepNumber)
        dispatch(setStep(stepNumber));
    };

    return (
        <div className="flex items-center justify-between mb-8">
        {steps.map((step, index) => (
            <React.Fragment key={step.number}>
            <div className="flex flex-col items-center">
                <div
                onClick={() => handleStepClick(step.number)}
                className={`w-12 h-12 rounded-full flex items-center justify-center font-semibold text-lg transition-all duration-300 cursor-pointer ${
                    step.number < currentStep
                    ? 'bg-green-500 text-white hover:bg-green-600 hover:scale-110 hover:shadow-lg'
                    : step.number === currentStep
                    ? 'bg-blue-500 text-white hover:bg-blue-600 hover:scale-110 hover:shadow-lg hover:rotate-6'
                    : 'bg-gray-300 text-gray-600 hover:bg-gray-400 hover:scale-105 hover:shadow-md'
                } active:scale-95`}
                >
                {step.number < currentStep ? '✓' : step.number}
                </div>
                <span 
                onClick={() => handleStepClick(step.number)}
                className={`mt-2 text-sm font-medium cursor-pointer transition-all duration-200 ${
                step.number <= currentStep ? 'text-gray-900 hover:text-blue-600 hover:scale-105' : 'text-gray-500 hover:text-gray-700 hover:scale-105'
                }`}>
                {step.label}
                </span>
            </div>
            {index < steps.length - 1 && (
                <div className="flex-1 h-1 mx-4 -mt-5">
                <div className={`h-full transition-all duration-300 ${
                    step.number < currentStep ? 'bg-green-500' : 'bg-gray-300'
                }`} />
                </div>
            )}
            </React.Fragment>
        ))}
        </div>
    );
};

export default ProgressStepper;