# Supabase Edge Functions

This directory contains Supabase Edge Functions (serverless functions deployed to Supabase).

## Structure

- Each subdirectory represents a single edge function
- Functions are deployed to `https://<project-id>.supabase.co/functions/v1/<function-name>`

## Development

To create a new function:
```bash
supabase functions new <function-name>
```

To test locally:
```bash
supabase start
supabase functions serve
```

To deploy:
```bash
supabase functions deploy <function-name>
```

## Documentation

See [Supabase Edge Functions Documentation](https://supabase.com/docs/guides/functions)
