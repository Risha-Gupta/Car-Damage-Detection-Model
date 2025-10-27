import React from 'react';

const ProgressStepper = ({ currentStep }) => {
    const steps = [
        { number: 1, label: 'Stage 1' },
        { number: 2, label: 'Stage 2' },
        { number: 3, label: 'Stage 3' },
        { number: 4, label: 'Stage 4' },
        { number: 5, label: 'Stage 5' }
    ];

    return (
        <div className="flex items-center justify-between mb-8">
        {steps.map((step, index) => (
            <React.Fragment key={step.number}>
            <div className="flex flex-col items-center">
                <div
                className={`w-12 h-12 rounded-full flex items-center justify-center font-semibold text-lg transition-colors ${
                    step.number < currentStep
                    ? 'bg-green-500 text-white'
                    : step.number === currentStep
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-300 text-gray-600'
                }`}
                >
                {step.number < currentStep ? '✓' : step.number}
                </div>
                <span className={`mt-2 text-sm font-medium ${
                step.number <= currentStep ? 'text-gray-900' : 'text-gray-500'
                }`}>
                {step.label}
                </span>
            </div>
            {index < steps.length - 1 && (
                <div className="flex-1 h-1 mx-4 mt-[-20px]">
                <div className={`h-full transition-colors ${
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