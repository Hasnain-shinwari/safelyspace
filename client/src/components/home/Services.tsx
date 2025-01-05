import { useEffect, useState } from "react";
import ServiceCard from "./ServiceCard";
import { FaArrowRight } from "react-icons/fa";
import { MdContentCopy } from "react-icons/md";
import { LuScanLine, LuFlower } from "react-icons/lu";
import { GoShield } from "react-icons/go";

const iconMap = {
  MdContentCopy: MdContentCopy,
  LuScanLine: LuScanLine,
  LuFlower: LuFlower,
  GoShield: GoShield,
};

const Services = () => {
  const [serviceCardData, setServiceCardData] = useState([]);

  useEffect(() => {
    fetch("/data/serviceCardData.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch data");
        }
        return response.json();
      })
      .then((data) => setServiceCardData(data))
      .catch((error) => console.error(error));
  }, []);
  return (
    <div className="mt-[95px]">
      <p className="text-3xl">
        <span className="font-bold text-nowrap">Safely Space</span> provides you
        with a comprehensive set of features, specifically{" "}
        <span className="font-bold italic">designed</span> to help you work with
        and effectively remove violence from{" "}
        <span className="font-thin italic">your videos</span>.
      </p>
      <ServiceCard serviceCardData={serviceCardData} iconMap={iconMap} />
      <div className="my-20 text-3xl">
        <p>
          Trusted by <span className="font-bold italic">Thousands</span>
        </p>
        <p>The best at making sure you are safe</p>
      </div>
      <div className="grid grid-cols-4 gap-x-3">
        <div className="flex flex-col justify-between bg-white space-y-2 p-4 rounded-xl">
          <div className="space-y-3">
            <p className="text-6xl">
              27<span className="text-[#D2ED2D]">%</span>
            </p>
            <p className="text-lg">Traffic increase one week post-launch</p>
          </div>
          <div className="space-y-4">
            <hr className="border border-t" />
            <div className="flex items-center space-x-4">
              <p>Read Story</p>
              <FaArrowRight className="h-[13px] w-[13px]" />
            </div>
          </div>
        </div>
        <div className="flex flex-col justify-between bg-white space-y-2 p-4 rounded-xl">
          <div className="space-y-3">
            <p className="text-6xl">
              1.3M<span className="text-[#D2ED2D]">+</span>
            </p>
            <p className="text-lg">Views</p>
          </div>
          <div className="space-y-4">
            <hr className="border border-t" />
            <div className="flex items-center space-x-4">
              <p>Read Story</p>
              <FaArrowRight className="h-[13px] w-[13px]" />
            </div>
          </div>
        </div>
        <div className="flex flex-col justify-between bg-white space-y-2 p-4 rounded-xl">
          <div className="space-y-3">
            <p className="text-6xl">
              200<span className="text-[#D2ED2D]">+</span>
            </p>
            <p className="text-lg">Bluepen site launched</p>
          </div>
          <div className="space-y-4">
            <hr className="border border-t" />
            <div className="flex items-center space-x-4">
              <p>Read Story</p>
              <FaArrowRight className="h-[13px] w-[13px]" />
            </div>
          </div>
        </div>
        <div className="flex flex-col justify-between bg-white space-y-2 p-4 rounded-xl">
          <div className="space-y-3">
            <p className="text-6xl">
              3<span className="text-[#D2ED2D]">x</span>
            </p>
            <p className="text-lg">Faster time to launcg</p>
          </div>
          <div className="space-y-4">
            <hr className="border border-t" />
            <div className="flex items-center space-x-4">
              <p>Read Story</p>
              <FaArrowRight className="h-[13px] w-[13px]" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Services;
