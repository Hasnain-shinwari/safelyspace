const Info = () => {
  return (
    <div className="flex flex-col items-center mt-20">
      <div className="w-fit flex flex-col items-center">
        <h3 className="text-3xl mb-10">A platform designed for you</h3>
        <div className="text-2xl space-y-5">
          <p>
            <span className="font-semibold">01</span> Lightning-Fast Speed
          </p>
          <hr className="border border-t border-[#C2C2C2]" />
          <p>
            <span className="font-semibold">02</span> User-Friendly UI
          </p>
          <hr className="border border-t border-[#C2C2C2]" />
          <p>
            <span className="font-semibold">03</span>Robust ML Models Used for
            Detection
          </p>
        </div>
        <button className="bg-black text-white px-6 py-3 mt-10 rounded-full text-lg font-semibold">Start Building</button>
      </div>
    </div>
  );
};

export default Info;
