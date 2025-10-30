import axios from "axios";
import { PIPELINE_API_BASE_URL } from "../../public/constants";

export const getIsDamaged = (IMAGE) => {
    const formData = new FormData();
    formData.append('image', IMAGE); 
    
    return axios.post(PIPELINE_API_BASE_URL + "/predict", formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    });
}