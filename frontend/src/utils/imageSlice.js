import { createSlice } from "@reduxjs/toolkit";

const imageSlice = createSlice({
    name: "image",
    initialState: {
        image: null,
        step: 0,
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
    },
});

export const { setImage, increment, decrement } = imageSlice.actions;
export default imageSlice.reducer;
