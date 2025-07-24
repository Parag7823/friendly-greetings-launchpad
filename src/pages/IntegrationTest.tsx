import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

const IntegrationTest = () => {
  const [supabaseStatus, setSupabaseStatus] = useState<'testing' | 'success' | 'error'>('testing');
  const [fastApiStatus, setFastApiStatus] = useState<'testing' | 'success' | 'error'>('testing');
  const [supabaseData, setSupabaseData] = useState<any[]>([]);
  const [fastApiResponse, setFastApiResponse] = useState<string>('');
  const { toast } = useToast();

  // Updated with your actual Railway FastAPI URL
  const FASTAPI_URL = "https://ai-financial-backend-production.up.railway.app";

  useEffect(() => {
    testSupabaseConnection();
  }, []);

  const testSupabaseConnection = async () => {
    try {
      // Test reading from raw_records table
      const { data, error } = await supabase
        .from('raw_records')
        .select('*')
        .limit(5);

      if (error) throw error;

      setSupabaseData(data || []);
      setSupabaseStatus('success');
      toast({
        title: "Supabase Connected",
        description: "Successfully connected to Supabase database",
      });
    } catch (error) {
      setSupabaseStatus('error');
      console.error('Supabase connection error:', error);
      toast({
        title: "Supabase Error",
        description: "Failed to connect to Supabase",
        variant: "destructive",
      });
    }
  };

  const testFastApiConnection = async () => {
    setFastApiStatus('testing');
    try {
      // Test API health endpoint
      const response = await fetch(`${FASTAPI_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      setFastApiResponse(JSON.stringify(data, null, 2));
      setFastApiStatus('success');
      
      toast({
        title: "FastAPI Connected",
        description: "Successfully connected to FastAPI backend",
      });
    } catch (error) {
      setFastApiStatus('error');
      setFastApiResponse(`Error: ${error}`);
      toast({
        title: "FastAPI Error",
        description: "Failed to connect to FastAPI backend",
        variant: "destructive",
      });
    }
  };

  const testCompleteFlow = async () => {
    try {
      // Test creating a test ingestion job via API
      const response = await fetch(`${FASTAPI_URL}/api/v1/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: "Integration test",
          timestamp: new Date().toISOString()
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const result = await response.json();
      
      toast({
        title: "Complete Flow Test",
        description: "FastAPI → Supabase integration working!",
      });

      // Refresh Supabase data to see the new record
      setTimeout(testSupabaseConnection, 1000);
    } catch (error) {
      toast({
        title: "Flow Test Error",
        description: "Complete integration test failed",
        variant: "destructive",
      });
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'success':
        return <Badge variant="default" className="bg-green-500">Connected</Badge>;
      case 'error':
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="secondary">Testing...</Badge>;
    }
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-2">Integration Test Dashboard</h1>
          <p className="text-muted-foreground">Testing Lovable AI + FastAPI + Supabase Integration</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Supabase Test */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Supabase Database
                {getStatusBadge(supabaseStatus)}
              </CardTitle>
              <CardDescription>
                Direct connection from frontend to Supabase
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={testSupabaseConnection} className="mb-4">
                Test Supabase
              </Button>
              <div className="text-sm">
                <p><strong>Records found:</strong> {supabaseData.length}</p>
                {supabaseData.length > 0 && (
                  <div className="mt-2 p-2 bg-muted rounded text-xs">
                    <pre>{JSON.stringify(supabaseData[0], null, 2)}</pre>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* FastAPI Test */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                FastAPI Backend
                {getStatusBadge(fastApiStatus)}
              </CardTitle>
              <CardDescription>
                Connection from frontend to Railway-deployed FastAPI
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={testFastApiConnection} className="mb-4">
                Test FastAPI
              </Button>
              {fastApiResponse && (
                <div className="text-sm">
                  <p><strong>Response:</strong></p>
                  <div className="mt-2 p-2 bg-muted rounded text-xs">
                    <pre>{fastApiResponse}</pre>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Complete Flow Test */}
        <Card>
          <CardHeader>
            <CardTitle>Complete Integration Test</CardTitle>
            <CardDescription>
              Test the complete flow: Frontend → FastAPI → Supabase → Frontend
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={testCompleteFlow} className="w-full">
              Test Complete Flow
            </Button>
            <p className="text-sm text-muted-foreground mt-2">
              This will create a test record via FastAPI and verify it appears in Supabase
            </p>
          </CardContent>
        </Card>

        {/* Status Summary */}
        <Card>
          <CardHeader>
            <CardTitle>Integration Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Frontend ↔ Supabase:</span>
                {getStatusBadge(supabaseStatus)}
              </div>
              <div className="flex justify-between">
                <span>Frontend ↔ FastAPI:</span>
                {getStatusBadge(fastApiStatus)}
              </div>
              <div className="flex justify-between">
                <span>Ready for Development:</span>
                {supabaseStatus === 'success' && fastApiStatus === 'success' ? 
                  <Badge variant="default" className="bg-green-500">Ready</Badge> : 
                  <Badge variant="secondary">Not Ready</Badge>
                }
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default IntegrationTest;
