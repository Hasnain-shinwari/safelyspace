import { GoPaperclip } from "react-icons/go";
import { IoSearchOutline } from "react-icons/io5";

const Hero = () => {
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
        <div className="flex justify-center mt-[80px]">
          <div className="flex items-center space-x-14 bg-[#B17979] px-10 py-5 rounded-full">
            <GoPaperclip className="w-[22px] h-[22px]" />
            <p className="text-lg text-white">
              Please provide a link to your video or attach the video file.
            </p>
            <IoSearchOutline className="w-[22px] h-[22px]" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
