import './App.css'
import MapView from './components/MapView'

export default function App() {
  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <nav style={{ background: '#0F6E56', color: 'white', padding: '12px 20px', fontSize: '18px', fontWeight: 500 }}>
        ⛵ ChainGuard
      </nav>
      <div style={{ flex: 1 }}>
        <MapView />
      </div>
    </div>
  )
}