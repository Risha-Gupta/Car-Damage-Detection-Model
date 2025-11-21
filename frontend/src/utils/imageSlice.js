import { createSlice } from "@reduxjs/toolkit";

const imageSlice = createSlice({
    name: "image",
    initialState: {
        image: null,
        step: 1,
        finished: false,
        stepStatus: {
            1: "idle",      
            2: "idle",
            3: "idle",
            4: "idle",
            5: "idle"
        }
    },

    reducers: {

        setImage: (state, action) => {
            state.image = action.payload;
        },

        increment: (state) => {
            if (state.step < 5) {
                state.step += 1;
                if (state.step === 5) {
                    state.finished = true;
                }
            }
        },

        decrement: (state) => {
            if (state.step > 1) {
                state.step -= 1;
                if (state.finished && state.step < 5) {
                    state.finished = false;
                }
            }
        },

        setStep: (state, action) => {
            state.step = action.payload;
        },
        setStepStatus: (state, action) => {
            const { step, status } = action.payload;
            state.stepStatus[step] = status;
        },

        resetPipeline: (state) => {
            state.step = 1;
            state.finished = false;
            state.stepStatus = {
                1: "idle",
                2: "idle",
                3: "idle",
                4: "idle",
                5: "idle"
            };
        }
    },
});

export const { 
    setImage, 
    increment, 
    decrement, 
    setStep,
    setStepStatus,
    resetPipeline 
} = imageSlice.actions;

export default imageSlice.reducer;
