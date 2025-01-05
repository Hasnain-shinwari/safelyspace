import { Link } from "react-router-dom";
const Navbar = () => {
  return (
    <nav className="flex justify-between text-lg py-7">
      <h1 className="font-semibold">SafelySpace</h1>
      <div className="flex space-x-12">
        <Link to="/">Home</Link>
        <Link to={"/about"}>About</Link>
        <Link to="/contact">Contact</Link>
      </div>
      <button className="bg-black text-white px-4 py-2 rounded-full">
        Get Started
      </button>
    </nav>
  );
};

export default Navbar;
