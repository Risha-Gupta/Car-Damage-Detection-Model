import ProgressStepper from "./ProgressStepper";
const Header = ({currentStep,setCurrentStep}) => {
    return (
        <div className="w-full max-w-4xl mx-auto p-8">
            <ProgressStepper currentStep={currentStep} setCurrentStep={setCurrentStep}/>
        </div>
    )
}
export default Header;