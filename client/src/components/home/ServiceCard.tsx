import React from "react";

interface ServiceCardInterface {
  icon: string;
  title: string;
  text: string;
}

interface ServiceCardProps {
  serviceCardData: ServiceCardInterface[];
  iconMap: { [key: string]: React.ComponentType };
}

const ServiceCard: React.FC<ServiceCardProps> = ({
  serviceCardData,
  iconMap,
}) => {
  return (
    <div className="grid grid-cols-4 gap-6 p-6">
      {serviceCardData.map((card, index) => {
        const IconComponent = iconMap[card.icon];
        return (
          <div key={index} className="flex flex-col justify-between space-y-8 p-4 rounded-lg">
            <div>
              <div className="text-5xl bg-[#D9D9D9] mb-6 p-5 w-fit rounded-lg">
                {IconComponent && <IconComponent />}
              </div>
              <h3 className="text-2xl font-semibold">{card.title}</h3>
            </div>
            <p className="text-gray-600">{card.text}</p>
          </div>
        );
      })}
    </div>
  );
};

export default ServiceCard;
