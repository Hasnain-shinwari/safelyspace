import { useState } from "react";
import { GoPaperclip } from "react-icons/go";
import { IoSearchOutline } from "react-icons/io5";
import { Dialog } from "@headlessui/react";
import axios from "axios";

const Hero = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [videoLink, setVideoLink] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleUpload = async () => {
    if (!videoLink && !videoFile) {
      alert("Please provide a video link or upload a file.");
      return;
    }

    setIsSubmitting(true);

    console.log("videoLink:", videoLink);
    
    const formData = new FormData();
    if (videoLink) formData.append("videoLink", videoLink);
    if (videoFile) formData.append("videoFile", videoFile);

    try {
      const response = await axios.post("http://localhost:8000/api/upload-video", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        withCredentials: true,
      });
      alert(
        "Video uploaded successfully! Metadata: " +
          JSON.stringify(response.data)
      );
    } catch (error) {
      console.error("Error uploading video:", error);
      alert("Failed to upload video. Please try again.");
    } finally {
      setIsSubmitting(false);
      setIsModalOpen(false);
    }
  };

  return (
    <div className="mt-[80px]">
      <div className="flex justify-between text-3xl space-x-8">
        <p className="w-[600px]">
          Ensuring Safe Content for Children with AI-Powered Detection
        </p>
        <p className="mt-20 w-[600px]">
          Detect Violent and Harmful Content in Videos
        </p>
      </div>
      <div>
        <div
          className="flex justify-center mt-[80px] cursor-pointer"
          onClick={() => setIsModalOpen(true)}
        >
          <div className="flex items-center space-x-14 bg-[#B17979] px-10 py-5 rounded-full">
            <GoPaperclip className="w-[22px] h-[22px]" />
            <p className="text-lg text-white">
              Please provide a link to your video or attach the video file.
            </p>
            <IoSearchOutline className="w-[22px] h-[22px]" />
          </div>
        </div>
      </div>

      {/* Modal */}
      {isModalOpen && (
        <Dialog
          open={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          className="fixed inset-0 z-10 flex items-center justify-center bg-black bg-opacity-50"
        >
          <div className="bg-white rounded-lg p-6 w-[500px]">
            <Dialog.Title className="text-2xl font-bold mb-4">
              Upload Video
            </Dialog.Title>
            <div className="space-y-4">
              <div>
                <label
                  className="block text-sm font-medium mb-2"
                  htmlFor="videoLink"
                >
                  Video Link
                </label>
                <input
                  id="videoLink"
                  type="text"
                  placeholder="Enter YouTube link"
                  value={videoLink}
                  onChange={(e) => setVideoLink(e.target.value)}
                  className="w-full px-4 py-2 border rounded-md"
                />
              </div>
              <div>
                <label
                  className="block text-sm font-medium mb-2"
                  htmlFor="videoFile"
                >
                  Upload File
                </label>
                <input
                  id="videoFile"
                  type="file"
                  accept="video/*"
                  onChange={(e) =>
                    e.target.files && setVideoFile(e.target.files[0])
                  }
                  className="w-full px-4 py-2 border rounded-md"
                />
              </div>
              <div className="flex justify-end space-x-4">
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 bg-gray-200 rounded-md"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpload}
                  className={`px-4 py-2 bg-blue-600 text-white rounded-md ${
                    isSubmitting ? "opacity-50 cursor-not-allowed" : ""
                  }`}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? "Uploading..." : "Submit"}
                </button>
              </div>
            </div>
          </div>
        </Dialog>
      )}
    </div>
  );
};

export default Hero;
