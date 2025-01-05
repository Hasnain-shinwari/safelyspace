import React from "react";

const ContactForm = () => {
	return (
		<div>
			<form action="" className="flex flex-col mb-6">
				<div className="flex justify-between mb-6">
					<div className="flex flex-col">
						<label htmlFor="FullName">Full Name</label>
						<input
							type="text"
							id="FullName"
							placeholder="Dr. Muhammad Nawaz"
							className="py-2 pr-44 pl-2 border border-gray-500 bg-transparent mb-6"
						/>
						<label htmlFor="Dropdown">What are you interested in?</label>
						<select name="Interested" id="Dropdown" className="py-2 pr-44 pl-2 border border-gray-500 bg-transparent">
							<option value="performance">Related to Performance</option>
							<option value="detection">ML isn't detecting properly.</option>
							<option value="attachment">I can't attach the files.</option>
						</select>
					</div>
					<div className="flex flex-col">
						<label htmlFor="Email">Email</label>
						<input
							type="text"
							id="Email"
							placeholder="name@example.com"
							className="py-2 pr-44 pl-2 border border-gray-500 bg-transparent mb-6"
						/>
						<label htmlFor="issues">Issue you are facing</label>
						<select name="Interested" id="Dropdown" className="py-2 pr-44 pl-2 border border-gray-500 bg-transparent">
							<option value="performance">Related to Performance</option>
							<option value="detection">ML isn't detecting properly.</option>
							<option value="attachment">I can't attach the files.</option>
						</select>
					</div>
				</div>
				<div className="">
					<form action="" className="flex flex-col">
						<label htmlFor="message">Message</label>
						<input
							type="text"
							placeholder="Type your message here"
							className="pl-2 pt-2 pb-44 pr-72 border border-gray-500 bg-transparent"
						/>
					</form>
				</div>
			</form>
		</div>
	);
};

export default ContactForm;
