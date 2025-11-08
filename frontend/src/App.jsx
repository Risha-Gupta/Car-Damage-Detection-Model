import './App.css'
import Header from "./components/Header"
import Body from './components/Body'
import { useState } from 'react';
function App() {
  const [currentStep, setCurrentStep] = useState(1);
  return (
    <>
      <Header currentStep={currentStep} setCurrentStep={setCurrentStep}/>
      <Body currentStep={currentStep} setCurrentStep={setCurrentStep}/>
    </>
  )
}

export default App
