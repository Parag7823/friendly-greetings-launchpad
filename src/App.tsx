import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/components/AuthProvider";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import Index from "./pages/Index";
import IntegrationTest from "./pages/IntegrationTest";
import NotFound from "./pages/NotFound";
import Integrations from "./pages/Integrations";
import SyncHistory from "./pages/SyncHistory";

const queryClient = new QueryClient();

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/chat" element={<Index />} />
              <Route path="/connectors" element={<Integrations />} />
              <Route path="/connectors/:connectionId/history" element={<SyncHistory />} />
              <Route path="/upload" element={<Index />} />
              <Route path="/integrations" element={<Integrations />} />
              <Route path="/test" element={<IntegrationTest />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
