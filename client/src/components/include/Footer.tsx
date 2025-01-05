const Footer = () => {
  return (
    <footer className="space-y-28 pb-5 mt-32">
      <div className="flex justify-between">
        <div className="space-y-4">
          <h2 className="font-semibold text-5xl">safelyspace</h2>
          <p className="w-[250px]">
            Build and launch sites quickly- and safely — with powerful features
            designed to help large teams collaborate.
          </p>
        </div>
        <div className="flex mt-5 space-x-14">
          <div className="space-y-2">
            <h4 className="text-lg font-semibold">About</h4>
            <p>About Us</p>
            <p>Our Mission</p>
          </div>
          <div className="space-y-2">
            <h4 className="text-lg font-semibold">Features</h4>
            <p>Video Analytics</p>
            <p>AI Powered Insights</p>
            <p>Reporting Tools</p>
            <p>Multimodal Content Analysis</p>
          </div>
          <div className="space-y-2">
            <h4 className="text-lg font-semibold">Legal</h4>
            <p>Privacy Policy</p>
            <p>Terms of Services</p>
            <p>Content Moderation Guidelines</p>
          </div>
          <div className="space-y-2">
            <h4 className="text-lg font-semibold">Support</h4>
            <p>Contact Us</p>
            <p>Help Center</p>
            <p>Feedbacl Form</p>
            <p>Report and Issue</p>
          </div>
        </div>
      </div>
      <div className="flex justify-between">
        <p>@ Copyright 2024 safelyspace. all rights reserved.</p>
        <div className="flex space-x-5">
          <p>Privacy Policy</p>
          <p>Terms of Uses</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
