import './App.css'
import Header from "./components/Header"
import Body from './components/Body'
import MainHeader from './components/MainHeader';
import { useState } from 'react';
function App() {
  const [currentStep, setCurrentStep] = useState(1);
  return (
    <>
      <MainHeader/>
      <div className='flex'>
        <Header currentStep={currentStep} setCurrentStep={setCurrentStep} className='w-64 shrink-0'/>
        <Body currentStep={currentStep} setCurrentStep={setCurrentStep} className='flex-1'/>
      </div>
    </>
  )
}

export default App
