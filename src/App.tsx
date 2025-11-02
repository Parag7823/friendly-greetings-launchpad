import React, { Suspense } from "react";
const ShadToaster = React.lazy(() => import("@/components/ui/toaster").then(m => ({ default: m.Toaster })));
const SonnerToaster = React.lazy(() => import("@/components/ui/sonner").then(m => ({ default: m.Toaster })));
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/components/AuthProvider";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import Index from "./pages/Index";
import IntegrationTest from "./pages/IntegrationTest";
import NotFound from "./pages/NotFound";
import SyncHistory from "./pages/SyncHistory";

const queryClient = new QueryClient();

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <TooltipProvider>
          <Suspense fallback={null}>
            <ShadToaster />
          </Suspense>
          <Suspense fallback={null}>
            <SonnerToaster />
          </Suspense>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/chat" element={<Index />} />
              <Route path="/connectors/:connectionId/history" element={<SyncHistory />} />
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
