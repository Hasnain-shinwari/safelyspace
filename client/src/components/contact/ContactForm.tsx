import React from 'react'

const ContactForm = () => {
  return (
    <div>
      <form action="" className='flex mb-6'>
        <div className="flex flex-col mr-16">
          <label htmlFor="FullName">Full Name</label>
          <input type="text" id='FullName' placeholder='Muhammad Ali' className='py-2 pr-64 pl-2 border border-gray-500 bg-transparent' />
        </div>
        <div className="flex flex-col">
          <label htmlFor="Email">Email</label>
          <input type="text" id='Email' placeholder='name@example.com' className='py-2 pr-64 pl-2 border border-gray-500 bg-transparent'/>
        </div>
      </form>
      <form action="" className='flex'>
      <div className="flex flex-col mr-16">
          <label htmlFor="Dropdown">What are you interested in?</label>
          <select name="cars" id="Dropdown">
            <option value="volvo">Volvo</option>
            <option value="saab">Saab</option>
            <option value="mercedes">Mercedes</option>
            <option value="audi">Audi</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label htmlFor="Email">Email</label>
          <input type="text" id='Email' placeholder='name@example.com' className='py-2 pr-64 pl-2 border border-gray-500 bg-transparent'/>
        </div>
      </form>
    </div>
  )
}

export default ContactForm