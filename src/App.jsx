import './App.css'
import MapView from './components/MapView'
import { mockShipments } from './data/mockShipments'

function getRiskBand(score) {
  if (score >= 80) return 'high'
  if (score >= 40) return 'medium'
  return 'low'
}

export default function App() {
  const completedSteps = [
    'React project ready',
    'Leaflet.js installed',
    'Basic layout skeleton created',
    'World map rendered',
    'Mock shipment routes plotted',
  ]

  const highRiskShipments = mockShipments.filter((shipment) => shipment.risk >= 80).length
  const mediumRiskShipments = mockShipments.filter(
    (shipment) => shipment.risk >= 40 && shipment.risk < 80,
  ).length
  const lowRiskShipments = mockShipments.filter((shipment) => shipment.risk < 40).length

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Supply Chain Visibility Demo</p>
          <h1>ChainGuard global shipment map</h1>
        </div>
        <div className="status-pill">Live mock network</div>
      </header>

      <section className="hero-grid">
        <article className="hero-card hero-card-primary">
          <p className="section-label">Dashboard status</p>
          <h2>Requested setup checklist is complete.</h2>
          <p className="hero-copy">
            The app now shows a responsive React layout, a Leaflet-powered world map, and mock
            shipment corridors drawn across major logistics lanes.
          </p>
        </article>

        <article className="hero-card metrics-card">
          <p className="section-label">Route overview</p>
          <div className="metrics-grid">
            <div>
              <span className="metric-value">{mockShipments.length}</span>
              <span className="metric-label">Active routes</span>
            </div>
            <div>
              <span className="metric-value">{highRiskShipments}</span>
              <span className="metric-label">High risk</span>
            </div>
            <div>
              <span className="metric-value">{mediumRiskShipments}</span>
              <span className="metric-label">Medium risk</span>
            </div>
            <div>
              <span className="metric-value">{lowRiskShipments}</span>
              <span className="metric-label">Low risk</span>
            </div>
          </div>
        </article>
      </section>

      <main className="content-grid">
        <section className="panel map-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">World map</p>
              <h3>Shipment routes by lane risk</h3>
            </div>
            <div className="legend">
              <span className="legend-item">
                <span className="legend-dot legend-high" />
                High
              </span>
              <span className="legend-item">
                <span className="legend-dot legend-medium" />
                Medium
              </span>
              <span className="legend-item">
                <span className="legend-dot legend-low" />
                Low
              </span>
            </div>
          </div>

          <div className="map-frame">
            <MapView shipments={mockShipments} />
          </div>
        </section>

        <aside className="sidebar">
          <section className="panel">
            <div className="panel-heading compact">
              <div>
                <p className="section-label">Checklist</p>
                <h3>Completed scope</h3>
              </div>
            </div>

            <ul className="checklist">
              {completedSteps.map((step) => (
                <li key={step} className="checklist-item">
                  <span className="check-icon">OK</span>
                  <span>{step}</span>
                </li>
              ))}
            </ul>
          </section>

          <section className="panel">
            <div className="panel-heading compact">
              <div>
                <p className="section-label">Mock shipments</p>
                <h3>Route snapshots</h3>
              </div>
            </div>

            <div className="shipment-list">
              {mockShipments.map((shipment) => (
                <article key={shipment.id} className="shipment-card">
                  <div className="shipment-row">
                    <strong>{shipment.id}</strong>
                    <span className={`risk-badge risk-${getRiskBand(shipment.risk)}`}>
                      {getRiskBand(shipment.risk)} risk
                    </span>
                  </div>
                  <p className="shipment-route">
                    {shipment.originLabel} to {shipment.destinationLabel}
                  </p>
                  <p className="shipment-meta">{shipment.cargo}</p>
                  <p className="shipment-meta">
                    {shipment.stage} - ETA {shipment.eta}
                  </p>
                </article>
              ))}
            </div>
          </section>
        </aside>
      </main>
    </div>
  )
}
