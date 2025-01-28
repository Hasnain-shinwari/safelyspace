const ContactForm = () => {
  return (
    <div className="mt-[50px]">
      <div className="text-4xl font-semibold mb-10">
        <h2>Love to hear from you,</h2>
        <h2>Get in touch 👋</h2>
      </div>
      <form action="" className="flex flex-col mb-6">
        <div className="flex justify-between space-x-7 mb-6">
          <div className="flex flex-col w-full">
            <div>
              <label htmlFor="FullName" className="font-medium text-lg">
                Full Name
              </label>
              <input
                type="text"
                id="FullName"
                placeholder="Dr. Muhammad Nawaz"
                className="py-2 px-2 border border-gray-500 bg-transparent mb-6 w-full mt-2"
              />
            </div>
            <div>
              <label htmlFor="Dropdown" className="font-medium text-lg">
                What are you interested in?
              </label>
              <select
                name="Interested"
                id="Dropdown"
                className="py-[10px] px-2 border border-gray-500 bg-transparent w-full mt-2"
              >
                <option value="performance">Related to Performance</option>
                <option value="detection">ML isn't detecting properly.</option>
                <option value="attachment">I can't attach the files.</option>
              </select>
            </div>
          </div>
          <div className="flex flex-col w-full">
            <div>
              <label htmlFor="Email" className="font-medium text-lg">
                Email
              </label>
              <input
                type="text"
                id="Email"
                placeholder="name@example.com"
                className="px-2 py-2 border border-gray-500 bg-transparent mb-6 w-full mt-2"
              />
            </div>
            <div>
              <label htmlFor="issues" className="font-medium text-lg">
                Issue you are facing
              </label>
              <select
                name="Interested"
                id="Dropdown"
                className="py-[10px] px-2 border border-gray-500 bg-transparent w-full mt-2"
              >
                <option value="performance">Related to Performance</option>
                <option value="detection">ML isn't detecting properly.</option>
                <option value="attachment">I can't attach the files.</option>
              </select>
            </div>
          </div>
        </div>
        <div>
          <div className="flex flex-col">
            <label htmlFor="message" className="font-medium text-lg">
              Message
            </label>
            <input
              type="text"
              placeholder="Type your message here"
              className="pl-2 pt-2 pb-44 pr-72 border border-gray-500 bg-transparent mt-2"
            />
          </div>
          <button className="py-2 px-36 bg-black mt-10 rounded-xl text-white">
            Just Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ContactForm;
