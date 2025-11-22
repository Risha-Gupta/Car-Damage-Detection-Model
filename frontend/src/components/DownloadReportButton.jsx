import { Download } from "lucide-react";
import axios from "axios";
import { PIPELINE_API_BASE_URL } from "../../public/constants";

const DownloadReportButton = ({ downloadUrl }) => {
    if (!downloadUrl) return null;

    const handleDownload = async () => {
        try {
            let fullUrl;
            
            if (downloadUrl.startsWith("http")) {
                // Already a full URL
                fullUrl = downloadUrl;
            } else if (downloadUrl.startsWith("/api")) {
                // URL already has /api prefix, use base host only
                fullUrl = `http://localhost:8000${downloadUrl}`;
            } else {
                // Relative path, append to base URL
                fullUrl = `${PIPELINE_API_BASE_URL}${downloadUrl}`;
            }

            const response = await axios.get(fullUrl, {
                responseType: "blob"
            });

            // Create a download link
            const url = window.URL.createObjectURL(response.data);
            const link = document.createElement("a");
            link.href = url;
            
            // Extract filename from URL or use default
            const filename = downloadUrl.split("/").pop() || "insurance_report.pdf";
            link.download = filename;
            
            // Trigger download
            document.body.appendChild(link);
            link.click();
            
            // Cleanup
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Download failed:", error);
            alert("Failed to download report. Please try again.");
        }
    };

    return (
        <button
            onClick={handleDownload}
            className="
                inline-flex items-center gap-2
                bg-indigo-600 hover:bg-indigo-700
                text-white font-medium
                px-4 py-3 rounded-lg
                shadow-md transition
                cursor-pointer
            "
        >
            <Download size={20} />
            Download Insurance Report
        </button>
    );
};

export default DownloadReportButton;