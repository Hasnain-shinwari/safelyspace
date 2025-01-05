import Navbar from "./components/include/Navbar";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import About from "./pages/About";
import Contact from "./pages/Contact";
import Footer from "./components/include/Footer";

function App() {
  return (
    <main className="bg-[#E2E3E9]">
      <section className="container">
        <BrowserRouter>
          <Navbar />
          <Routes>
            <Route index element={<Home />} />
            <Route path="about" element={<About />} />
            <Route path="contact" element={<Contact />} />
          </Routes>
          <Footer />
        </BrowserRouter>
      </section>
    </main>
  );
}

export default App;
