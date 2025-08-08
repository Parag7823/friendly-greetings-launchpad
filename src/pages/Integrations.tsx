import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { FileSpreadsheet, Server } from "lucide-react";
import IntegrationCard from "@/components/IntegrationCard";

const Integrations = () => {
  const navigate = useNavigate();

  useEffect(() => {
    document.title = "Integrations — Finley AI";
  }, []);

  const handleExcelAction = () => {
    navigate("/");
    // allow routing to settle then open the existing upload flow in the sidebar
    setTimeout(() => {
      window.dispatchEvent(new Event("open-excel-upload"));
    }, 0);
  };

  const hasConnections = false; // no fetching per spec; assumed empty state

  return (
    <main className="w-full">
      {hasConnections === false && (
        <div className="px-6 pt-6">
          <div className="finley-card rounded-md border border-border bg-card text-card-foreground p-4">
            Connect integrations to automate your financial workflows.
          </div>
        </div>
      )}

      <section className="p-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <IntegrationCard
            icon={<FileSpreadsheet className="w-8 h-8 text-green-600" aria-hidden />}
            title="Excel Integration"
            description="Upload and process spreadsheets directly from Excel or CSV files."
            actionLabel="Connect / Upload"
            onAction={handleExcelAction}
          />

          <IntegrationCard
            icon={<Server className="w-8 h-8 text-muted-foreground" aria-hidden />}
            title="Embedded Infrastructure"
            description="Direct connections to bank feeds, ERPs, and financial systems."
            statusLabel="Coming Soon"
            disabled
          />
        </div>
      </section>
    </main>
  );
};

export default Integrations;
