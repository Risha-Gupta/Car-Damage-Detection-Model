import axios from "axios";
import { PIPELINE_API_BASE_URL } from "../../public/constants";

export const getIsDamaged = (IMAGE) => {
    const formData = new FormData();
    formData.append('file', IMAGE); 
    return axios.post(PIPELINE_API_BASE_URL + "/predict", formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    });
}
export const getDamagedAreas = (IMAGE) => {
    const formData = new FormData();
    formData.append('file', IMAGE); 
    return axios.post(PIPELINE_API_BASE_URL + "/locate", formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    });
}
export const getDamageTypes = (IMAGE) => {
    const formData = new FormData();
    formData.append('file', IMAGE); 
    return axios.post(PIPELINE_API_BASE_URL + "/segment", formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    });
}
export const classifyDamage = (IMAGE) => {
    const formData = new FormData();
    formData.append('file', IMAGE); 
    return axios.post(PIPELINE_API_BASE_URL + "/classify", formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    });
}
export const estimateCost = (IMAGE) => {
    const formData = new FormData();
    formData.append('file', IMAGE); 
    return axios.post(PIPELINE_API_BASE_URL + "/estimate", formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    });
}

export const downloadReport = (downloadUrl) => {
    const fullUrl = downloadUrl.startsWith("http") 
        ? downloadUrl 
        : `${PIPELINE_API_BASE_URL}${downloadUrl}`;
    
    return axios.get(fullUrl, {
        responseType: "blob"
    });
};