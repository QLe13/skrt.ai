import NextAuth from "next-auth"
import GoogleProvider from "next-auth/providers/google"
import db from "../../../utils/db"

const handler = NextAuth({
    secret: 'N39+eJnNYk7eI6YDQOtHCibBrPoR+HiVHPp8/pQuiX8=',
    providers: [
        GoogleProvider({
          clientId: process.env.GOOGLE_CLIENT_ID as string,
          clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
        })
      ],
    callbacks: {
        async signIn({ user, account, profile, email, credentials }) {
          const userEmail = profile?.email
            if (userEmail) {
                const [rows] = await db.query('SELECT * FROM users WHERE email = ? LIMIT 1', [userEmail]);
                if (Array.isArray(rows) && rows.length === 0) {
                    // create new user
                    const [result] = await db.query('INSERT INTO users (email, credits) VALUES (?, ?)', [userEmail,999]);
                    if (result) {
                        return true
                    } else {
                        return false
                    }
                }
                else {
                    return true
                }
            } else {
                return false
            }
        },

    }
})

export { handler as GET, handler as POST }