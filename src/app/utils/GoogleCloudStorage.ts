import { Storage } from "@google-cloud/storage";

// Ensure your private key replaces any escaped newlines from environment variables
// so that it is correctly formatted for JSON parsing
const privateKey = process.env.Cloud_Storage_private_key?.replace(/\\n/g, '\n')
// credentials: {
//     type: 'service_account',
//     project_id: 'xxxxxxx',
//     private_key_id: 'xxxx',
//     private_key:'-----BEGIN PRIVATE KEY-----xxxxxxx\n-----END PRIVATE KEY-----\n',
//     client_email: 'xxxx',
//     client_id: 'xxx',
//     auth_uri: 'https://accounts.google.com/o/oauth2/auth',
//     token_uri: 'https://oauth2.googleapis.com/token',
//     auth_provider_x509_cert_url: 'https://www.googleapis.com/oauth2/v1/certs',
//     client_x509_cert_url: 'xxx',
//     }
// });
const key = {
    projectId: process.env.Cloud_Storage_project_id,
    credentials: {
        type: process.env.Cloud_Storage_type,
        project_id: process.env.Cloud_Storage_project_id,
        private_key_id: process.env.Cloud_Storage_private_key_id,
        private_key: privateKey,
        client_email: process.env.Cloud_Storage_client_email,
        client_id: process.env.Cloud_Storage_client_id,
        auth_uri: process.env.Cloud_Storage_auth_uri,
        token_uri: process.env.Cloud_Storage_token_uri,
        auth_provider_x509_cert_url: process.env.Cloud_Storage_auth_provider_x509_cert_url,
        client_x509_cert_url: process.env.Cloud_Storage_client_x509_cert_url,
    }
};
const storage = new Storage(key);



export default storage;