import ProgressStepper from "./ProgressStepper";

const Header = ({ currentStep, setCurrentStep }) => {
    return (
        <div className="h-screen shrink-0 bg-gray-50 border-r border-gray-200">
            <ProgressStepper currentStep={currentStep} setCurrentStep={setCurrentStep}/>
        </div>
    )
}

export default Header;