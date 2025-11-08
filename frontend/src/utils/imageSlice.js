import { createSlice } from "@reduxjs/toolkit";

const imageSlice = createSlice({
    name: "image",
    initialState: {
        image: null,
        step: 1,
        finished: false,
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
            if (state.step > 0) {
                state.step -= 1;
                if (state.finished && state.step < 5) {
                    state.finished = false;
                }
            }
        },
        setStep : (state,action) => {
            state.step = action.payload
        }
    },
});

export const { setImage, increment, decrement , setStep} = imageSlice.actions;
export default imageSlice.reducer;
