import {configureStore} from "@reduxjs/toolkit";
import imageReducer from "./imageSlice";
const appStore = configureStore({
        reducer:{
            image:imageReducer
        },
});
export default appStore;