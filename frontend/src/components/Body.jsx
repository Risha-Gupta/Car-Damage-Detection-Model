/* eslint-disable no-unused-vars */
import { useDispatch, useSelector } from "react-redux";
import { increment, decrement } from "../utils/imageSlice";
import Predictor from "./Predictor"
import Locator from "./Locator";
const Body = ({currentStep, setCurrentStep}) => {
    const step = useSelector((state) => state.image.step);
    const dispatch = useDispatch();
    const handleNextStage = () => {
        setCurrentStep(step+1)
        dispatch(increment());
    };

    const handlePrevStage = () => {
        setCurrentStep(step-1)
        dispatch(decrement());
    };
    const renderStage = () => {
        switch (step) {
            case 1:
                return <Predictor onNext={handleNextStage} />;
            case 2:
                return <Locator onBack={handlePrevStage} onNext={handleNextStage} />;
            default:
                return <Predictor onNext={handleNextStage} />;
        }
    };
    return (
        <div>
            {renderStage()}
        </div>
    )
}
export default Body;